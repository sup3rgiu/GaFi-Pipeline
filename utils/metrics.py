import tensorflow as tf
import tensorflow_probability as tfp
import keras
import types
import functools

tfk = tf.keras
tfkl = tf.keras.layers

# Borg Singleton to share attributes among subclassed metrics
class SharedMetric(tfk.metrics.Metric):
    _shared_borg_state = {}

    def __new__(cls, *args, **kwargs):
        obj = super(SharedMetric, cls).__new__(cls, *args, **kwargs)
        obj.__dict__ = cls._shared_borg_state
        return obj
    
    def __init__(self, image_size, normalize, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.image_size = image_size
        self.normalize = normalize
        if not hasattr(self, 'net'):
            self.net = self._build_net()
            
    def _build_net(self):
        print('Building InceptionV3 net')
        # resolution of Inception Score measurement
        is_image_size = 299
        
        # a pretrained InceptionV3 is used with its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining    
                   
        scale = 255.0/2 if self.normalize else 255.0
        offset = 255.0/2 if self.normalize else 0.0
        
        inception_net = tf.keras.applications.InceptionV3(
            include_top=True,
            input_shape=(is_image_size, is_image_size, 3),
            weights="imagenet",
            classifier_activation='softmax'
        )
        
        # Build a model that returns both logits and features, useful to have everything needed to compute FID,KID,IS in just one forward pass
        inception_net_split = tfk.Model(inception_net.input, [inception_net.output, inception_net.get_layer('avg_pool').output], name='inception_v3_split')
        
        inp = tfkl.Input((self.image_size, self.image_size, 3))
        x = tfkl.Rescaling(scale=scale, offset=offset)(inp)
        x = tfkl.Resizing(height=is_image_size, width=is_image_size)(x)
        x = tfkl.Lambda(tf.keras.applications.inception_v3.preprocess_input)(x)
        out = inception_net_split(x)
        return tfk.Model(inp, out, name='inception_net')
       
    def _shuffle_tensor(self, tensor):
        tensor = tf.random.shuffle(tensor)
        # We can use the stateless version to shuffle in the same way at each call even if in @tf.function
        #tensor = tf.random.experimental.stateless_shuffle(tensor, seed=(42,50))
        return tensor
        
    def update_state(self):
        pass
    
    def result(self):
        pass
    
    def reset_state(self):
        pass

class FID(SharedMetric):
    def __init__(self, image_size, normalize, name="fid", **kwargs):
        super().__init__(image_size=image_size, normalize=normalize, name=name, **kwargs)

        # FID is estimated per batch and is averaged across batches
        self.fid_tracker = tfk.metrics.Mean()
        
        # Keep these in memory for efficiency
        self.m = None
        self.sigma = None

    def _symmetric_matrix_square_root(self, mat, eps=1e-10):
        """ Compute square root of a symmetric matrix """
        # Unlike numpy, tensorflow's return order is (s, u, v)
        s, u, v = tf.linalg.svd(mat)
        # sqrt is unstable around 0, just use 0 in such case
        si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
        # Note that the v returned by Tensorflow is v = V
        # (when referencing the equation A = U S V^T)
        # This is unlike Numpy which returns v = V^T
        return tf.matmul(tf.matmul(u, tf.linalg.tensor_diag(si)), v, transpose_b=True)
    
    def _trace_sqrt_product(self, sigma, sigma_v):
        """ Find the trace of the positive sqrt of product of covariance matrices """
        # Note sqrt_sigma is called "A" in the proof above
        sqrt_sigma = self._symmetric_matrix_square_root(sigma)
        # This is sqrt(A sigma_v A) above
        sqrt_a_sigmav_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_v, sqrt_sigma))
        return tf.linalg.trace(self._symmetric_matrix_square_root(sqrt_a_sigmav_a))

    def update_state(self, real_images, generated_images):
        # Get stats of real images
        if self.m is None or self.sigma is None:
            #print('Generating FID m and sigma (should be here only once)')
            real_features = getattr(self, 'real_features', None)
            if real_features is None:
                print('Generating real features FID (should not be here)')
                _, real_features = self.net.predict(real_images, batch_size=128, verbose=0)
            real_features = tf.cast(real_features, tf.float64)
            self.m = tf.reduce_mean(real_features, axis=0)
            # Calculate the unbiased covariance matrix of real activations
            num_examples_real = tf.cast(tf.shape(real_features)[0], tf.float64)
            self.sigma = num_examples_real / (num_examples_real - 1) * tfp.stats.covariance(real_features) # np.cov(real_features, rowvar=False)
            
        # Get stats of generated images
        generated_features = getattr(self, 'generated_features', None)
        if generated_features is None:
            print('Generating fake features FID')
            generated_images = self._shuffle_tensor(generated_images) # needed if generated_images is label-ordered and not already shuffled
            generated_images = generated_images[:tf.shape(real_images)[0]] # number of real_images == number of generated_images. Subsample here to avoid useless computations
            _, generated_features = self.net.predict(generated_images, batch_size=128, verbose=0)
        generated_features = self._shuffle_tensor(generated_features)
        generated_features = tf.cast(generated_features, tf.float64)
        generated_features = generated_features[:tf.shape(real_images)[0]]
        m_w = tf.reduce_mean(generated_features, axis=0)
        num_examples_generated = tf.cast(tf.shape(generated_features)[0], tf.float64)
        sigma_w = num_examples_generated / (num_examples_generated - 1) * tfp.stats.covariance(generated_features)

        def _calculate_fid(m, m_w, sigma, sigma_w):
            """Returns the Frechet distance given the sample mean and covariance."""
            # Find the Tr(sqrt(sigma sigma_w)) component of FID
            sqrt_trace_component = self._trace_sqrt_product(sigma, sigma_w)

            # Compute the two components of FID.

            # First the covariance component.
            # Here, note that trace(A + B) = trace(A) + trace(B)
            trace = tf.linalg.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

            # Next the distance between means.
            mean = tf.reduce_sum(tf.math.squared_difference(m, m_w))  # Equivalent to L2 but more stable.
            fid = trace + mean
            fid = tf.cast(fid, tf.float32)
            return fid

        result = _calculate_fid(self.m, m_w, self.sigma, sigma_w)
        #return result

        # update the average FID estimate
        self.fid_tracker.update_state(result)

    def result(self):
        return {'FID': self.fid_tracker.result()}

    def reset_state(self):
        self.fid_tracker.reset_state()

class KID(SharedMetric):
    def __init__(self, image_size, normalize, name="kid", **kwargs):
        super().__init__(image_size=image_size, normalize=normalize, name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = tfk.metrics.Mean()
       
        # Keep this in memeory for efficiency
        self.mean_kernel_real = None

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float64)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images):
        self.real_features = getattr(self, 'real_features', None)
        if self.real_features is None:
            print('Generating real features KID (should not be here)')
            _, self.real_features = self.net.predict(real_images, batch_size=128, verbose=0)
        self.real_features = tf.cast(self.real_features, tf.float64)
        
        generated_features = getattr(self, 'generated_features', None)
        if generated_features is None:
            print('Generating fake features FID')
            generated_images = self._shuffle_tensor(generated_images) # needed if generated_images is label-ordered and not already shuffled
            generated_images = generated_images[:tf.shape(real_images)[0]] # number of real_images == number of generated_images. Subsample here to avoid useless computations
            _, generated_features = self.net.predict(generated_images, batch_size=128, verbose=0)
        generated_features = self._shuffle_tensor(generated_features)
        generated_features = tf.cast(generated_features, tf.float64)
        generated_features = generated_features[:tf.shape(real_images)[0]]

        # compute polynomial kernels using the two sets of features
        if self.mean_kernel_real is None:
            kernel_real = self.polynomial_kernel(self.real_features, self.real_features)
        kernel_generated = self.polynomial_kernel(generated_features, generated_features)
        kernel_cross = self.polynomial_kernel(self.real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(self.real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float64)
        if self.mean_kernel_real is None:
            #print('Generating mean_kernel_real KID (should be here only once)')
            self.mean_kernel_real = (tf.reduce_sum(kernel_real) - tf.linalg.trace(kernel_real)) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_generated = (tf.reduce_sum(kernel_generated) - tf.linalg.trace(kernel_generated)) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = self.mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross
        kid = tf.cast(kid, tf.float32)

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return {'KID': self.kid_tracker.result()}

    def reset_state(self):
        self.kid_tracker.reset_state()

class IS(SharedMetric):
    def __init__(self, image_size, normalize, name="is", **kwargs):
        super().__init__(image_size=image_size, normalize=normalize, name=name, **kwargs)
        
        # IS is estimated per batch and is averaged across batches
        self.is_mean_tracker = tfk.metrics.Mean(name='IS_mean')
        self.is_std_tracker = tfk.metrics.Mean(name='IS_std')
        
    def compute_is(self, p):
        q = tf.reduce_mean(p, axis=0)
        kl = tf.reduce_sum(p * (tf.math.log(p) - tf.math.log(q)), axis=1)
        #kl = tf.reduce_sum(p * (tf.nn.log_softmax(p_logits) - tf.math.log(q)), axis=1)
        log_score = tf.reduce_mean(kl)
        score = tf.exp(log_score)
        return score
        
    def update_state(self, generated_images, splits=0):
        self.g_probs = getattr(self, 'g_probs', None)
        if self.g_probs is None:
            print('Generating fake probs IS')
            self.g_probs, _ = self.net.predict(generated_images, batch_size=256, verbose=1)
        p = self.g_probs
        p = tf.cast(p, tf.float64) # Use maximum precision for best results.
        p = self._shuffle_tensor(p) # needed if generated_images is label-ordered and not already shuffled

        if splits > 0:
            scores = []
            splits = tf.split(p, splits)
            for i in range(tf.shape(splits)[0]):
                _p = tf.gather(splits, i)
                score = self.compute_is(_p)
                scores.append(score)
            score = tf.reduce_mean(scores)
            std = tf.math.reduce_std(scores)
        else:
            score = self.compute_is(p)
            std = 0.0
            
        score = tf.cast(score, tf.float32)

        # update the average IS estimate
        self.is_mean_tracker.update_state(score)
        self.is_std_tracker.update_state(std)

    def result(self):
        return {
            'IS_mean': self.is_mean_tracker.result(),
            'IS_std':  self.is_std_tracker.result(),
        }

    def reset_state(self):
        self.g_probs = None
        self.is_mean_tracker.reset_state()
        self.is_std_tracker.reset_state()

class MetricsWrapper(tfk.metrics.Metric):
    def __init__(self, metrics, name='metrics_wrapper', dtype=None, verbose=False, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.metrics_list = metrics
        self.verbose = verbose
        
    def update_state(self, real_images, generated_images, splits=0):
        for metric in self.metrics_list:
            if isinstance(metric, IS):
                self._possibly_set_shared_objects(metric, real_images=None, generated_images=generated_images)
                metric.update_state(generated_images, splits)
            elif isinstance(metric, (FID, KID)):
                self._possibly_set_shared_objects(metric, real_images=real_images, generated_images=generated_images)
                metric.update_state(real_images, generated_images)
            else:
                raise NotImplementedError(f"{metric.__class__.__name__} not implemented in the wrapper")
                
    def _possibly_set_shared_objects(self, metric, real_images, generated_images):
        if (real_images is not None and
            (getattr(metric, 'real_features', None) is None
             or getattr(metric, 'r_probs', None) is None)
           ):
            #print('Generating real stats')
            metric.r_probs, metric.real_features = metric.net.predict(real_images, batch_size=128, verbose=self.verbose)
        if (generated_images is not None and
            (getattr(metric, 'generated_features', None) is None
             or getattr(metric, 'g_probs', None) is None)
           ):
            #print('Generating fake stats')
            metric.g_probs, metric.generated_features = metric.net.predict(generated_images, batch_size=128, verbose=self.verbose)
    
    def result(self):
        result = {}
        for metric in self.metrics_list:
            _result = metric.result()
            if not isinstance(_result, dict):
                _result = {metric.name: _result}
            result.update(_result)
        return result
    
    def reset_state(self):
        for metric in self.metrics_list:
            # Reset variables related to generated images because they are different at each epoch
            metric.generated_features = None
            metric.g_probs = None
            metric.reset_state()
            
    def complete_reset(self):
        for metric in self.metrics_list:
            metric.generated_features = None
            metric.g_probs = None
            metric.real_features = None
            metric.r_probs = None
            metric.reset_state()

class NetDistance(keras.metrics.Metric):
    '''
     Args:
        net: The Network used to compute the output. If it is an instantiated object, it's used as is at every epoch,
             otherwise the network's weights are reset at the end of every epoch
        criterion: function used to compute the loss
    '''
    def __init__(self, net, criterion, name="classifier_distance", **kwargs):
        super().__init__(name=name, **kwargs)
        if isinstance(net, (type, types.FunctionType, functools.partial)):
            self.net_class = net
            self.net = net()
            self.should_reset_weights = True
        else:
            self.net = net
            self.should_reset_weights = False
            
        self.criterion = criterion

        self._build_trackers()
        
    def _build_trackers(self):
        net_output = self.net.output
        trackers = []
        if isinstance(net_output, dict):
            for k in net_output.keys():
                setattr(self, k, tfk.metrics.Mean(name=k))
                trackers.append(getattr(self, k))
        elif isinstance(net_output, list):
            for k in range(len(net_output)):
                tracker_name = f"{self.name}_{k}"
                setattr(self, tracker_name, tfk.metrics.Mean(name=tracker_name))
                trackers.append(getattr(self, tracker_name))
        else:
            trackers.append(tfk.metrics.Mean())
        self.trackers = trackers

    def update_state(self, real_x, generated_x, real_y=None, generated_y=None):
        real_output = self.net.predict(real_x, batch_size=128, verbose=0)
        generated_output = self.net.predict(generated_x, batch_size=128, verbose=0)
        
        if real_y == None and generated_y == None:
            cd = self.criterion(real_output, generated_output)
        else:
            cd = self.criterion(real_output, generated_output, real_y, generated_y)

        if len(self.trackers) == 1:
            self.trackers[0].update_state(cd)
        else:
            for i, tracker in enumerate(self.trackers):
                tracker.update_state(cd[i])

    def result(self):
        if len(self.trackers) == 1:
            return self.trackers[0].result()
        else:
            return {m.name+'_'+self.name: m.result() for m in self.trackers}

    def reset_state(self):
        for tracker in self.trackers:
            tracker.reset_state()
        if self.should_reset_weights:
            new_seed = tf.random.uniform((), maxval=10000, dtype=tf.int32)
            try:
                self.net = self.net_class(seed=new_seed)
            except:
                self.net = self.net_class()