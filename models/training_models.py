import tensorflow as tf
import keras
from utils.metrics import MetricsWrapper, NetDistance
import utils.misc as misc
import utils.losses as losses

tfk = tf.keras

# Custom tfk.Model to support eager execution of the test step
class CustomGANModel(tfk.Model):
    def __init__(self, num_inception_images=50000, run_val_eagerly=True, fixed_test_images=True, *args, **kwargs):
        super().__init__(*args, **kwargs)      
        self.num_inception_images = num_inception_images
        self.run_val_eagerly = run_val_eagerly
        self.fixed_test_images = fixed_test_images
        self.current_eval_epoch = 0
        
    def compile(self, val_metrics=[]):
        super().compile()
        self._val_metrics = val_metrics
              
    @property
    def val_metrics(self):
        return self._val_metrics
    
    @tf.function
    def train_step(self, data):
        raise NotImplementedError
    
    # ! We will run this function in eager mode because we will use .predict() (also inside the metrics)
    def test_step(self, data):
        self.current_eval_epoch += 1
        if self.config.RUN.compute_metrics and self.current_eval_epoch >= self.config.RUN.start_eval_epoch and len(self.val_metrics) > 0:
            x, y = data

            if hasattr(self, 'ema_generator'):
                generator = self.ema_generator
            else:
                generator = self.generator

            if self.fixed_test_images:
                z = tf.random.stateless_normal((self.num_inception_images, self.latent_dim), seed=(10,90))
                gen_y = tf.random.stateless_uniform((self.num_inception_images,), maxval=self.num_classes, dtype=tf.int32, seed=(10,90))
            else:
                z = tf.random.normal((self.num_inception_images, self.latent_dim))
                gen_y = tf.random.uniform((self.num_inception_images,), maxval=self.num_classes, dtype=tf.int32)

            if tf.rank(y) > 1:
                y = tf.argmax(y, -1)

            fake_dataset = generator.predict([z, gen_y], verbose=0, batch_size=128)

            # Compute metrics
            for m in self.val_metrics:
                if isinstance(m, NetDistance):
                    balanced_subsample_x, subsample_gen_y = misc.subsample_balanced_dataset(fake_dataset, gen_y, size=tf.shape(x)[0], return_labels=True) # Here the images are ordered by label. Shuffle them if needed
                    m.update_state(real_x=x, generated_x=balanced_subsample_x, real_y=y, generated_y=subsample_gen_y)
                elif isinstance(m, MetricsWrapper):
                    m.update_state(x, fake_dataset, splits=0)
                else:
                    raise NotImplementedError(f"Logic for metric {m.__class__.__name__} currently not implemented")

            # Build output dict with results of all metrics
            result = {}
            for m in self.val_metrics:
                _result = m.result()
                if not isinstance(_result, dict):
                    _result = {m.name: _result}
                result.update(_result)
            return result
        else:
            return {}
      
    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return self.test_function

        def step_function(model, iterator):
            """Runs a single evaluation step."""

            def run_step(data):
                outputs = model.test_step(data)
                # Ensure counter is updated only if `test_step` succeeds.
                with tf.control_dependencies(keras.engine.training._minimum_control_deps(outputs)):
                    model._test_counter.assign_add(1)
                return outputs

            if self._jit_compile:
                run_step = tf.function(
                    run_step, jit_compile=True, reduce_retracing=True
                )

            data = next(iterator)
#             outputs = model.distribute_strategy.run(run_step, args=(data,))
#             outputs = keras.engine.training.reduce_per_replica(
#                 outputs,
#                 self.distribute_strategy,
#                 reduction='auto',
#             )
            outputs = run_step(data)
            return outputs

        # Special case if steps_per_execution is one.
        if (
            self._steps_per_execution is None
            or self._steps_per_execution.numpy().item() == 1
        ):
            def test_function(iterator):
                """Runs a test execution with a single step."""
                return step_function(self, iterator)

            if not self.run_val_eagerly:
                test_function = tf.function(
                    test_function, reduce_retracing=True
                )

            if self._cluster_coordinator:
                self.test_function = (
                    lambda it: self._cluster_coordinator.schedule(
                        test_function, args=(it,)
                    )
                )
            else:
                self.test_function = test_function

        else:
            def test_function(iterator):
                """Runs a test execution with multiple steps."""
                for _ in tf.range(self._steps_per_execution):
                    outputs = step_function(self, iterator)
                return outputs

            if not self.run_val_eagerly:
                test_function = tf.function(
                    test_function, reduce_retracing=True
                )
            self.test_function = test_function

        return self.test_function
    
class GAN(CustomGANModel):
    def __init__(self, generator, discriminator, classifier_pretrained, cfgs, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.classifier = classifier_pretrained
        self.latent_dim = cfgs.MODEL.latent_dim
        self.d_updates_per_step = cfgs.OPTIMIZATION.d_updates_per_step
        self.split_batch_d_steps = cfgs.OPTIMIZATION.split_batch_d_steps
        self.random_labels = cfgs.OPTIMIZATION.random_labels
        self.num_classes = cfgs.DATA.num_classes
        self.compute_classifier_stats = cfgs.RUN.compute_classifier_stats
        self.save_optimizer = cfgs.OPTIMIZATION.save_optimizer
        self.config = cfgs
        
        if cfgs.MODEL.apply_g_ema:
            self.ema_decay = cfgs.MODEL.g_ema_decay or 0.9999
            self.itrs = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False,
                synchronization=tf.VariableSynchronization.ON_READ,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            )
            self.ema_start = cfgs.MODEL.g_ema_start or 0
            self.ema_generator = tfk.models.clone_model(self.generator)
            self.ema_generator.set_weights(self.generator.get_weights())

    def compile(self, g_optimizer, d_optimizer, **kwargs):
        super().compile(**kwargs)
        
        # Compile the models to set the optimizers.
        # In this way, if mixed-precision is used, the optimizers will be automatically wrapped with tf.keras.mixed_precision.LossScaleOptimizer
        self.generator.compile(optimizer=g_optimizer)
        self.discriminator.compile(optimizer=d_optimizer)
        
        self.D_loss_real_metric = tfk.metrics.Mean(name="D_loss_real")
        self.D_loss_fake_metric = tfk.metrics.Mean(name="D_loss_fake")
        self.G_loss_metric = tfk.metrics.Mean(name="G_loss")
        
        if self.config.DATA.categorical:
            acc_metric = tfk.metrics.CategoricalAccuracy
        else:
            acc_metric = tfk.metrics.SparseCategoricalAccuracy
        
        if self.compute_classifier_stats:
            self.classification_accuracy_real_tracker = acc_metric(name="c_acc_real")
            self.classification_accuracy_fake_tracker = acc_metric(name="c_acc_fake")
        if self.config.LOSS.grad_penalty_type != None:
            self.grad_penalty_tracker = tfk.metrics.Mean(name="grad_penalty")
        
        self.dis_loss = losses.discriminator_loss(loss_type=self.config.LOSS.type)
        self.gen_loss = losses.generator_loss(loss_type=self.config.LOSS.type)
        
    @property
    def metrics(self):
        metrics = [self.D_loss_fake_metric, self.D_loss_real_metric, self.G_loss_metric]
        if self.compute_classifier_stats:
            metrics += [self.classification_accuracy_real_tracker, self.classification_accuracy_fake_tracker]
        if self.config.LOSS.grad_penalty_type != None:
            metrics += [self.grad_penalty_tracker]
        metrics += self.val_metrics
        return metrics
    
    def call(self, data):
        return self.train_step(data)

    @tf.function
    def train_step(self, data):
        if self.config.MODEL.apply_g_ema: self.itrs.assign_add(1)

        real_images, labels = data
        basket_size = batch_size = tf.shape(real_images)[0]

        if self.split_batch_d_steps:
            real_images = tf.split(real_images, basket_size // self.config.DATA.batch_size)
            labels = tf.split(labels, basket_size // self.config.DATA.batch_size)
            batch_size = tf.shape(real_images[0])[0]        
        

        ### UPDATE DISCRIMINATOR ###
        D_loss_real_tot = 0
        D_loss_fake_tot = 0
        # Using tf.range would result in faster graph compilation time, but slower execution time (https://lukewood.xyz/blog/to-unroll-or-to-not-unroll)
        for step_index in range(self.d_updates_per_step):
            with tf.GradientTape(persistent=False) as tape:

                # Real inputs
                if self.split_batch_d_steps:
                    _x = real_images[step_index]
                    _y = labels[step_index]
                else:
                    _x = real_images
                    _y = labels

                # Fake inputs
                z_ = tf.random.normal(shape=(batch_size, self.latent_dim))
                if self.random_labels:
                    y_ = tf.cast(tf.random.uniform((batch_size, 1), minval=0, maxval=self.num_classes, dtype=tf.int32), _y.dtype)
                else:
                    y_ = _y
                G_z = self.generator([z_, y_], training=True)

                if self.config.LOSS.grad_penalty_type == 'r1':
                    D_fake = self.discriminator([G_z, y_], training=True)
                    D_real, grad_penalty = losses.r1_gradient_penalty(
                        discriminator=self.discriminator,
                        inputs=[_x, _y],
                        penalty_cost=self.config.LOSS.grad_penalty_cost)
                elif self.config.LOSS.grad_penalty_type is None:
                    # concatenate real data and the Generator's output along the batch dimension for improved efficiency.
                    D_input = tf.concat([G_z, _x], 0)
                    D_class = tf.concat([y_, _y], 0)
                    D_out = self.discriminator([D_input, D_class], training=True)
                    D_fake, D_real = tf.split(D_out, [tf.shape(G_z)[0], tf.shape(_x)[0]], axis=0)
                    grad_penalty = 0.0
                else:
                    raise ValueError(f"{self.config.LOSS.grad_penalty_type} is not a recognized gradient penalty type")
                
                D_loss_real, D_loss_fake = self.dis_loss(D_fake, D_real)

                D_loss = (D_loss_real + D_loss_fake) + grad_penalty
                D_loss_real_tot += D_loss_real
                D_loss_fake_tot += D_loss_fake

            # Using directly model.optimizer.minimize() will automatically scale the loss and unscale the gradients when mixed_precision is used.
            # Alternatively, we could use the following code (reference: https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer):
            # scaled_loss = self.discriminator.optimizer.get_scaled_loss(D_loss) # inside the tape!
            # grads = tape.gradient(scaled_loss, self.discriminator.trainable_weights)
            # grads = self.discriminator.optimizer.get_unscaled_gradients(grads)
            # self.discriminator.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
            self.discriminator.optimizer.minimize(D_loss, self.discriminator.trainable_weights, tape=tape)
        

        ### UPDATE GENERATOR ###
        with tf.GradientTape(persistent=False) as tape:
            z_ = tf.random.normal(shape=(batch_size, self.latent_dim))
            if self.random_labels:
                y_ = tf.cast(tf.random.uniform((batch_size, 1), minval=0, maxval=self.num_classes, dtype=tf.int32), _y.dtype)
            else:
                if self.split_batch_d_steps:
                    y_ = labels[step_index] # use the same labels as the batch of the last step of the discriminator
                else:
                    y_ = labels

            G_z = self.generator([z_, y_], training=True)

            D_input = G_z
            D_class = y_

            D_fake = self.discriminator([D_input, D_class], training=True)

            G_loss = self.gen_loss(D_fake)
            
            # Computer classifier stats (if needed)
            if self.compute_classifier_stats:
                predictions_real = self.classifier(real_images, training=False)
                predictions_fake = self.classifier(G_z, training=False)

        self.generator.optimizer.minimize(G_loss, self.generator.trainable_weights, tape=tape)
        

        ### UPDATE EMA GENERATOR (if needed) ###
        def update_ema():
            # track the exponential moving average of the generator's weights to decrease variance in the generation quality
            for weight, ema_weight in zip(self.generator.weights, self.ema_generator.weights):
                ema_weight.assign(self.ema_decay * ema_weight + (1 - self.ema_decay) * weight)
                
        def copy_weights():
            for weight, ema_weight in zip(self.generator.weights, self.ema_generator.weights):
                ema_weight.assign(weight)

        if self.config.MODEL.apply_g_ema:
            op = tf.cond(tf.equal(self.itrs, self.ema_start), true_fn=lambda: tf.print("\n\n*** Starting accumulating EMA ***\n"), false_fn=lambda: tf.no_op())
            op = tf.cond(tf.greater(self.itrs, self.ema_start), true_fn=update_ema, false_fn=copy_weights)
              

        ### UPDATE METRICS ###
        self.D_loss_fake_metric.update_state(D_loss_fake)
        self.D_loss_real_metric.update_state(D_loss_real)
        self.G_loss_metric.update_state(G_loss)
        
        out_dict = {
            "D_loss_real": self.D_loss_real_metric.result(),
            "D_loss_fake": self.D_loss_fake_metric.result(),
            "G_loss": self.G_loss_metric.result(),
        }
        
        if self.compute_classifier_stats:
            self.classification_accuracy_real_tracker.update_state(labels, predictions_real)
            self.classification_accuracy_fake_tracker.update_state(y_, predictions_fake)
            out_dict.update({
                "c_acc_real": self.classification_accuracy_real_tracker.result(),
                "c_acc_fake": self.classification_accuracy_fake_tracker.result(),
            })
            
        out_dict.update({
            "lr_g": self.generator.optimizer.lr(self.generator.optimizer.iterations) if isinstance(self.generator.optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule) else self.generator.optimizer.lr,
            "lr_d": self.discriminator.optimizer.lr(self.discriminator.optimizer.iterations) if isinstance(self.discriminator.optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule) else self.discriminator.optimizer.lr,
        })
            
        return out_dict