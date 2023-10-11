import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Log all messages except INFO and WARNING
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) # To suppress 'WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op...' message when saving a model with custom ops

import tensorflow as tf
from utils import datasets, misc, config, train_fns
import utils.parser as parser_utils
import pandas as pd
import re
import glob

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
    ###       Load dataset and set variables       ###
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
    _, val_ds, test_ds = ds.load_dataset(merge_train_val=cfgs.DATA.merge_train_val, splits=['val', 'test'])
    if val_ds is None:
        val_ds = test_ds

    dataset_size = ds.X_train_len
    class_samples = cfgs.PIPELINE.class_samples or int(dataset_size / cfgs.DATA.num_classes)
    best_class_samples = cfgs.PIPELINE.best_class_samples or class_samples
    print(f"Fake datasets will have a size of {class_samples} samples per class")

    testing_dataset = test_ds # test_ds / val_ds

    if cfgs.RUN.mixed_precision:
        print("[i] Using mixed precision")
        tfk.mixed_precision.set_global_policy('mixed_float16')

    ### Load the pretrained classifier to use for filtering ###
    classifier_pretrained = misc.load_classifier(classifier_type=cfgs.CLS_PRE.name, dataset=cfgs.DATA.dataset, optimizer=cfgs.CLS_PRE.optimizer, random_erasing=cfgs.CLS_PRE.random_erasing,
                                                 padding=ds.padding, image_size=ds.image_size, resize=ds.resize, normalize=ds.normalize, extra_name=cfgs.CLS_PRE.extra_name, cls_save_path=cfgs.CLS_PRE.save_path)

    ### Build the base classifier names used for training the classifier and loading the results, respectively ###
    c_name = (f"{cfgs.MODEL.name.lower()}_{class_samples}classSamples_bs{cfgs.OPTIMIZATION.batch_size}_{cfgs.OPTIMIZATION.optimizer.lower()}\
{'_erasing' if cfgs.AUG.random_erasing else ''}{f'_{cfgs.RUN.extra_name}' if cfgs.RUN.extra_name != '' else ''}")
    c_best_name = (f"{cfgs.MODEL.name.lower()}_{best_class_samples}classSamples_bs{cfgs.OPTIMIZATION.batch_size}_{cfgs.OPTIMIZATION.optimizer.lower()}\
{'_erasing' if cfgs.AUG.random_erasing else ''}{'_oneGforEpoch' if cfgs.PIPELINE.one_gan_for_epoch else ''}{f'_{cfgs.PIPELINE.best_extra_name}' if cfgs.PIPELINE.best_extra_name != '' else ''}")
    base_save_path = os.path.join(cfgs.RUN.save_path, ds.dataset_name, 'MultiGAN', c_name)


    ##################################################
    ###           Load all the generators          ###
    ##################################################
    gen_configs = {}
    for gan_name in cfgs.PIPELINE.gan_names:
        base_gan_path = os.path.join(cfgs.PIPELINE.load_path, ds.dataset_name, gan_name)
        best_hyper_results_path = os.path.join(base_gan_path, 'pipeline', c_best_name, 'best_hyperparameters', 'results.csv') # load results from 'best hyperparameters' step
        best_hyper_results = pd.read_csv(best_hyper_results_path)
        best_ckpt = best_hyper_results.iloc[best_hyper_results['val_accuracy'].idxmax()]['checkpoint_name'] # get the best checkpoint from the results

        ckpt_epoch = int(re.search(r"ckpt(\d+)", best_ckpt).group(1))
        best_stddev = float(re.search(r"std(\d+(\.\d+)?)", best_ckpt).group(1))
        best_threshold = float(re.search(r"filter(\d+(\.\d+)?)", best_ckpt).group(1))
        gan_seed = int(re.search(r"seed(\d+)", gan_name).group(1))

        print(f"\n**** {gan_name} ****")
        print(f"Best checkpoint epoch is {ckpt_epoch} with sttdev {best_stddev} and threshold {best_threshold}")

        base_load_path = os.path.join(base_gan_path, 'checkpoints')
        gen_path = os.path.join(base_load_path, f"*epoch*{str(ckpt_epoch).zfill(3)}*")
        gen = misc.load_model(model_path=gen_path, verbose=True)

        gen_config = {'gen': gen, 'ckpt_epoch': ckpt_epoch, 'seed': gan_seed, 'stddev': best_stddev, 'threshold': best_threshold}
        gen_configs[gan_name] = gen_config


    #####################################################
    ###   Train classifier with multiple generators   ###
    #####################################################

    num_gen = len(gen_configs)
    misc.pretty_print(f"Starting training with {num_gen} generators", separator='#', spacing='\n\n')

    if cfgs.PIPELINE.apply_standing_stats:
        for i, (k,v) in enumerate(gen_configs.items()):
            print(f"Applying standing stats for Gen {i}...")
            gen = v['gen']
            gen = misc.apply_standing_statistics(generator=gen, standing_max_batch=cfgs.PIPELINE.standing_stats_bs, standing_step=cfgs.PIPELINE.standing_stats_bs,
                                                 latent_dim=gen.input[0].shape[1], num_classes=cfgs.DATA.num_classes, safe_copy=True, verbose=True)
            v['gen'] = gen

    result = train_fns.run_multigan_step(
        gen_configs=gen_configs,
        class_samples=class_samples,
        testing_dataset=testing_dataset,
        classifier_save_path=base_save_path,
        cfgs=cfgs,
        classifier_pretrained=classifier_pretrained,
        filter=True,
        fixed_dset=False,
        recycling_period=1, # full recycle (i.e. Accurate Pipeline)
        recycling_factor=1,
    )
    print("\n")

    print(f"At the end of the training, the MultiGAN model achieved an accuracy of: {max(result['history']['val_accuracy'])*100:.2f}%")

    ### End of pipeline ###
    misc.pretty_print(f"DONE", separator='#', spacing='\n\n')

def main():
    parser = parser_utils.prepare_multigan_parser()
    args = parser.parse_args()
    args_dict = parser_utils.split_args(args)
    cfgs = config.MultiGANConfiguration(args.cfg_file)
    cfgs.update_cfgs(args_dict)
    run(cfgs)

if __name__ == '__main__':
  main()