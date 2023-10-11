import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Log all messages except INFO and WARNING
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) # To suppress 'WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op...' message when saving a model with custom ops

import tensorflow as tf
from utils import datasets, misc, config, optimizers, augmentation, callbacks
import utils.parser as parser_utils
from models import classifiers
from datetime import datetime

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

    ds = datasets.Dataset(
       dataset_name=cfgs.DATA.dataset,
       batch_size=cfgs.OPTIMIZATION.batch_size, 
       val_batch_size=cfgs.OPTIMIZATION.val_batch_size,
       categorical=cfgs.DATA.categorical, 
       val_categorical=cfgs.DATA.val_categorical, 
       normalize=cfgs.DATA.normalize, 
       dequantize=cfgs.DATA.dequantize,
       horizontal_flip=cfgs.AUG.random_flip,
       resize=cfgs.DATA.resize,
       resize_size=cfgs.DATA.image_size, 
       padding=cfgs.DATA.padding,
       drop_remainder=cfgs.DATA.drop_remainder,
       cfgs=cfgs,
       seed=seed
    )

    # If merge_train_val is True, then train_ds is actually train+val datasets and val_ds is None
    # Otherwise train_ds, val_ds and test_ds are the actual splits
    train_ds, val_ds, test_ds = ds.load_dataset(merge_train_val=cfgs.DATA.merge_train_val, splits=['train', 'val', 'test'])
    if val_ds is None:
        val_ds = test_ds


    ##################################################
    ###     Create the optimizer and the model     ###
    ##################################################

    if cfgs.RUN.mixed_precision:
        print("[i] Using mixed precision")
        tfk.mixed_precision.set_global_policy('mixed_float16')

    opt = optimizers.get_optimizer(optimizer=cfgs.OPTIMIZATION.optimizer, lr=cfgs.OPTIMIZATION.lr, momentum=cfgs.OPTIMIZATION.momentum,
                                   nesterov=cfgs.OPTIMIZATION.nesterov, weight_decay=cfgs.OPTIMIZATION.weight_decay)
    
    input_shape = (ds.image_size, ds.image_size, ds.num_channels)
    augmentation_layer = augmentation.augment(image_size=ds.image_size, normalize=ds.normalize, random_flip=cfgs.AUG.random_flip, random_crop=cfgs.AUG.random_crop,
                                              random_rotation=cfgs.AUG.random_rotation, random_zoom=cfgs.AUG.random_zoom, random_erasing=cfgs.AUG.random_erasing, autocast=False)
    classifier = classifiers.get_classifier(model=cfgs.MODEL.name, input_shape=input_shape, num_classes=ds.num_classes, width=cfgs.MODEL.width,
                                            seed=seed, augmentation_layer=augmentation_layer)
    
    run_name = f"""pretrained_{classifier.name.lower()}_{ds.dataset_name.lower()}_{cfgs.OPTIMIZATION.optimizer}{f'_pad{ds.image_size}' if ds.padding else ''}\
{f'_resized{ds.image_size}' if ds.resize else ''}{'_normalized' if ds.normalize else ''}{'_erasing' if cfgs.AUG.random_erasing else ''}{f'_{cfgs.RUN.extra_name}' if cfgs.RUN.extra_name != '' else ''}"""

    print(f"Training \"{run_name}\"")


    ##################################################
    ###              Train the model               ###
    ##################################################

    loss = tfk.losses.CategoricalCrossentropy()
    classifier.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    save_path = os.path.join(cfgs.RUN.save_path, ds.dataset_name, run_name)
    if os.path.exists(save_path):
        print("Path already exists. Adding temp name to current path...")
        now = datetime.now()
        save_path = save_path + '_' + now.strftime("%Y%m%d%H%M%S")
        
    history = classifier.fit(
        x=train_ds,
        epochs=cfgs.OPTIMIZATION.epochs,
        validation_data=(val_ds),
        callbacks=[
            #tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', restore_best_weights=True, patience=10),
            tfk.callbacks.ModelCheckpoint(filepath=save_path, save_weights_only=False, monitor='val_accuracy', mode='max', save_best_only=True),
            #tfk.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001),
            #callbacks.LearningRateStepScheduler(callbacks.lr_scheduler(milestones=[32000, 48000], gamma=0.1)),
            callbacks.LearningRateEpochScheduler(callbacks.lr_scheduler(milestones=[60, 80], gamma=0.1)),
            tfk.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=['lr']), # manually add ProgbarLogger to decide its position w.r.t. other callbacks (otherwise it's always the last callback and so other callbacks' prints could interrupt the progbar during validation step)
            callbacks.LogBestAccuracyCallback(),
        ]
    ).history


    ############################################################
    ###     Evaluate the best checkpoint on the test set     ###
    ############################################################

    print(f"Loading best checkpoint from \"{save_path}\"")
    best_classifier = tfk.models.load_model(save_path)
    best_classifier.trainable = False

    results = best_classifier.evaluate(test_ds, return_dict=True, verbose=0)
    print(f"Test accuracy: {results['accuracy']:.4f}")

def main():
    parser = parser_utils.prepare_classifier_parser()
    args = parser.parse_args()
    args_dict = parser_utils.split_args(args)
    cfgs = config.ClassifierConfiguration(args.cfg_file)
    cfgs.update_cfgs(args_dict)
    run(cfgs)

if __name__ == '__main__':
  main()