import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Log all messages except INFO and WARNING
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) # To suppress 'WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op...' message when saving a model with custom ops

import tensorflow as tf
from utils import datasets, misc, config, optimizers, losses, train_fns
from utils.metrics import MetricsWrapper, KID, FID, IS, NetDistance
import utils.parser as parser_utils
import utils.callbacks
from models import classifiers, big_gan, training_models
import math
from functools import partial

import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

tfk = tf.keras

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

def run(cfgs):

    seed = cfgs.RUN.seed
    misc.set_seed(seed)

    ##################################################
    ###              Load the dataset              ###
    ##################################################

    basket_size = cfgs.OPTIMIZATION.batch_size * cfgs.OPTIMIZATION.d_updates_per_step * cfgs.OPTIMIZATION.acml_steps

    ds = datasets.Dataset(
       dataset_name=cfgs.DATA.dataset,
       batch_size=basket_size, 
       val_batch_size=cfgs.OPTIMIZATION.val_batch_size,
       categorical=cfgs.DATA.categorical, 
       val_categorical=cfgs.DATA.val_categorical, 
       normalize=cfgs.DATA.normalize, 
       dequantize=cfgs.DATA.dequantize,
       horizontal_flip=cfgs.DATA.horizontal_flip,
       resize=cfgs.DATA.resize,
       resize_size=cfgs.DATA.image_size, 
       padding=cfgs.DATA.padding,
       drop_remainder=cfgs.DATA.drop_remainder,
       cfgs=cfgs,
       seed=seed
    )

    # If merge_train_val is True, then train_ds is actually train+val datasets and val_ds is None
    # Otherwise train_ds, val_ds and test_ds are the actual splits
    train_ds, _, _ = ds.load_dataset(merge_train_val=cfgs.DATA.merge_train_val, splits=['train', 'val', 'test'])
    _, _, X_val, y_val, X_test, y_test = ds.get_numpy_splits()
    if X_val is None:
        X_val = X_test
        y_val = y_test

    cfgs.DATA.train_size = len(train_ds) * basket_size # actual train size, due to drop_remainder


    ###################################################
    ###   Create models, optimizers and callbacks   ###
    ###################################################

    if cfgs.RUN.mixed_precision:
        print("[i] Using mixed precision")
        c6320be55e06bd47a8796182f3993d6e27c03678902c9866

    classifier_pretrained = None
    if cfgs.RUN.compute_classifier_stats or cfgs.RUN.compute_metrics:
        classifier_pretrained = misc.load_classifier(classifier_type=cfgs.CLS_PRE.name, dataset=cfgs.DATA.dataset, optimizer=cfgs.CLS_PRE.optimizer, random_erasing=cfgs.CLS_PRE.random_erasing,
                                                     padding=ds.padding, image_size=ds.image_size, resize=ds.resize, normalize=ds.normalize, extra_name=cfgs.CLS_PRE.extra_name, cls_save_path=cfgs.CLS_PRE.save_path)

    generator = big_gan.get_generator(cfgs=cfgs, dataset=ds)
    discriminator = big_gan.get_discriminator(cfgs=cfgs, dataset=ds)
    
    g_optimizer = optimizers.get_optimizer(optimizer=cfgs.OPTIMIZATION.optimizer, lr=cfgs.OPTIMIZATION.g_lr,
                                           beta_1=cfgs.OPTIMIZATION.g_beta1, beta_2=cfgs.OPTIMIZATION.g_beta2, epsilon=cfgs.OPTIMIZATION.adam_eps)
    
    d_optimizer = optimizers.get_optimizer(optimizer=cfgs.OPTIMIZATION.optimizer, lr=cfgs.OPTIMIZATION.d_lr,
                                           beta_1=cfgs.OPTIMIZATION.d_beta1, beta_2=cfgs.OPTIMIZATION.d_beta2, epsilon=cfgs.OPTIMIZATION.adam_eps)
    
    callbacks = []
    val_metrics = []

    ### Build evaluation metrics (if needed) ###
    if cfgs.RUN.compute_metrics:
        input_shape = (ds.image_size, ds.image_size, ds.num_channels)
        classifier_random = classifiers.resnet20(input_shape=input_shape, num_classes=ds.num_classes, width=64)
        outputs = {'prob': classifier_random.output, 'feat': classifier_random.get_layer('flatten').output}
        classifier_random = tfk.Model(classifier_random.input, outputs)
        classifier_random = partial(tfk.models.clone_model, classifier_random)

        classi_pre = classifier_pretrained
        outputs = {'prob': classi_pre.output, 'feat': classi_pre.get_layer('flatten').output}
        classi_pre = tfk.Model(classi_pre.input, outputs)

        val_metrics = [
            MetricsWrapper(metrics=[KID(image_size=ds.image_size, normalize=ds.normalize),
                                    FID(image_size=ds.image_size, normalize=ds.normalize),
                                    IS(image_size=ds.image_size, normalize=ds.normalize)],
                           name='metrics_wrapper', verbose=False),
            NetDistance(classifier_random, criterion=losses.probs_and_features_distances, name='rand'),
            NetDistance(classi_pre, criterion=losses.probs_and_features_distances, name='pre'),
        ]
  
    ### Generate run name from config ###
    is_deep = 'deep' in cfgs.MODEL.name
    run_name = f"""BigGAN_{'Deep_' if is_deep else ''}{'StudioGAN_' if cfgs.MODEL.studiogan and is_deep else ''}\
{'concat_' if cfgs.MODEL.residual_concat else ''}{ds.dataset_name.lower()}{f'_pad{ds.image_size}' if ds.padding else ''}\
{f'_resized{ds.image_size}' if ds.resize else ''}{'_normalized' if ds.normalize else ''}\
_seed{cfgs.RUN.seed}{f'_{cfgs.OPTIMIZATION.d_updates_per_step}discStep'}{f'_{cfgs.RUN.extra_name}' if cfgs.RUN.extra_name != '' else ''}"""

    base_save_path = os.path.join(cfgs.RUN.save_path, ds.dataset_name)

    if not cfgs.RUN.reload:
        run_name = misc.check_and_fix_duplicate_run(run_name, base_save_path)
    
    ### Create checkpoint callback (if needed) ###
    if cfgs.RUN.save:
        sub_model_name = 'ema_generator' # model to checkpoint
        filepath = os.path.join(base_save_path, run_name, 'checkpoints', f'{sub_model_name}_epoch{{epoch:03d}}')

        # metrics_to_monitor = ['val_'+item for sublist in [list(m.result().keys()) for m in cmodel.val_metrics] for item in sublist]
        metrics_to_monitor = {
            'val_KID': 'min',
            'val_FID': 'min',
            'val_IS_mean': 'max',
            'val_prob_rand': 'min',
            'val_feat_rand': 'min',
            'val_prob_pre': 'min',
            'val_feat_pre': 'min',
        }

        checkpoint_cb = utils.callbacks.MultiModelCheckpoint(
            filepath = filepath,
            monitor = list(metrics_to_monitor.keys()),
            save_best_only = False, # True to save checkpoints only when at least one monitored metric improves
            save_weights_only = False,
            include_optimizer = cfgs.OPTIMIZATION.save_optimizer_ckpt,
            mode = list(metrics_to_monitor.values()),
            save_freq = "epoch",
            initial_value_threshold = None,
            sub_model_name = sub_model_name,
            verbose = 0
        )
        callbacks += [checkpoint_cb]

    ### Build the model with the training logic ###
    gan = training_models.GAN(generator=generator, discriminator=discriminator, classifier_pretrained=classifier_pretrained, cfgs=cfgs)
    gan.compile(g_optimizer=g_optimizer,
                d_optimizer=d_optimizer,
                val_metrics=val_metrics)


    ##################################################
    ###              Train the model               ###
    ##################################################

    epochs = cfgs.OPTIMIZATION.epochs or math.ceil(cfgs.OPTIMIZATION.total_steps / len(train_ds))

    train_fns.train_gan(
        gan=gan,
        training_dataset=train_ds,
        epochs=epochs,
        config=cfgs,
        steps_per_epoch=None,
        run_name=run_name,
        show_plots=cfgs.RUN.show_plots,
        plot_with_title=cfgs.RUN.plot_with_title,
        plot_title=cfgs.RUN.plot_title,
        callbacks=callbacks,
        validation_data=(X_val, y_val) if cfgs.RUN.compute_metrics else None,
        validation_batch_size=X_val.shape[0], # make validation_data as one big batch only
        save=cfgs.RUN.save,
        keep_images=cfgs.RUN.keep_images,
        sample_ema=True,
        apply_standing_stats=False,
        fixed_seed=cfgs.RUN.fixed_seed,
        reload=cfgs.RUN.reload,
    )

    misc.pretty_print(f"DONE", separator='#', spacing='\n\n')

def main():
    parser = parser_utils.prepare_gan_parser()
    args = parser.parse_args()
    args_dict = parser_utils.split_args(args)
    cfgs = config.GANConfiguration(args.cfg_file)
    cfgs.update_cfgs(args_dict)
    run(cfgs)

if __name__ == '__main__':
  main()