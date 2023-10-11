import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Log all messages except INFO and WARNING
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) # To suppress 'WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op...' message when saving a model with custom ops

import tensorflow as tf
from utils import datasets, misc, config, train_fns
import utils.parser as parser_utils
import pandas as pd
import re

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
    print(f"Fake datasets will have a size of {class_samples} samples per class")

    testing_dataset = test_ds # test_ds / val_ds

    steps = cfgs.PIPELINE.steps
    if isinstance(steps, str):
        steps = list(steps)

    if cfgs.RUN.mixed_precision:
        print("[i] Using mixed precision")
        tfk.mixed_precision.set_global_policy('mixed_float16')

    ### Load the pretrained classifier to use for filtering ###
    classifier_pretrained = misc.load_classifier(classifier_type=cfgs.CLS_PRE.name, dataset=cfgs.DATA.dataset, optimizer=cfgs.CLS_PRE.optimizer, random_erasing=cfgs.CLS_PRE.random_erasing,
                                                 padding=ds.padding, image_size=ds.image_size, resize=ds.resize, normalize=ds.normalize, extra_name=cfgs.CLS_PRE.extra_name, cls_save_path=cfgs.CLS_PRE.save_path)

    ### Build the base run name and the save and load paths ###
    c_name = (f"{cfgs.MODEL.name.lower()}_{class_samples}classSamples_bs{cfgs.OPTIMIZATION.batch_size}_{cfgs.OPTIMIZATION.optimizer.lower()}\
{'_erasing' if cfgs.AUG.random_erasing else ''}{f'_{cfgs.RUN.extra_name}' if cfgs.RUN.extra_name != '' else ''}")

    base_save_path = os.path.join(cfgs.RUN.save_path, ds.dataset_name, cfgs.PIPELINE.gan_name, 'pipeline', c_name)
    base_load_path = os.path.join(cfgs.PIPELINE.load_path, ds.dataset_name, cfgs.PIPELINE.gan_name, 'checkpoints')


    ##################################################
    ###       STEP 1: Checkpoint optimization      ###
    ##################################################

    if 'all' in steps or 'ckpt' in steps:
        current_step = 'ckpt_optimization'
        misc.pretty_print(f"Running step: '{current_step}'", separator='#', spacing='\n\n')

        ckpt_epochs = misc.get_list_or_tuple(cfgs.PIPELINE.ckpt_epochs, type_check=int)
        for ckpt_epoch in ckpt_epochs:

            gen_path = os.path.join(base_load_path, f"*epoch*{str(ckpt_epoch).zfill(3)}*")

            gen = misc.load_model(model_path=gen_path, verbose=True)
            if gen is None:
                print(f"Generator checkpoint corresponding to epoch {ckpt_epoch} not found, skipping...")
                continue
            
            if cfgs.PIPELINE.apply_standing_stats:
                gen = misc.apply_standing_statistics(generator=gen, standing_max_batch=cfgs.PIPELINE.standing_stats_bs, standing_step=cfgs.PIPELINE.standing_stats_bs,
                                                     latent_dim=gen.input[0].shape[1], num_classes=cfgs.DATA.num_classes, safe_copy=True, verbose=True)
            train_fns.run_pipeline_step(
                generator=gen,
                class_samples=class_samples,
                testing_dataset=testing_dataset,
                classifier_save_path=os.path.join(base_save_path, current_step, f'ckpt{str(ckpt_epoch).zfill(3)}'),
                cfgs=cfgs,
                classifier_pretrained=classifier_pretrained,
                filter=True,
                threshold=0.0,
                stddev=1.0,
                fixed_dset=False,
                recycling_period=10, # Fast Pipeline
                recycling_factor=1,
            )
            print("\n")
        

    ##################################################
    ###         STEP 2: Stddev optimization        ###
    ##################################################

    if 'all' in steps or 'stddev' in steps:
        current_step = 'stddev_optimization'
        misc.pretty_print(f"Running step: '{current_step}'", separator='#', spacing='\n\n')

        results_path = os.path.join(base_save_path, 'ckpt_optimization', 'results.csv') # load results from previous step
        ckpt_results = pd.read_csv(results_path)
        # TODO: Currently, the best checkpoint is determined by the highest validation accuracy in the results.csv file, regardless of the DS (recycle period)
        # or r (recycling factor) values. Since these values are currently hardcoded to 10 and 1, respectively, this approach is acceptable.
        # However, if these values become customizable in the future, the best checkpoint selection process must take them into account as well.
        best_ckpt = ckpt_results.iloc[ckpt_results['val_accuracy'].idxmax()]['checkpoint_name'] # get the best checkpoint from the previous step

        print("Best checkpoint is: ", best_ckpt)

        ckpt_epoch = int(re.search(r"ckpt(\d+)", best_ckpt).group(1))
        gen_path = os.path.join(base_load_path, f"*epoch*{str(ckpt_epoch).zfill(3)}*")
        gen = misc.load_model(model_path=gen_path, verbose=True)

        if cfgs.PIPELINE.apply_standing_stats:
            gen = misc.apply_standing_statistics(generator=gen, standing_max_batch=cfgs.PIPELINE.standing_stats_bs, standing_step=cfgs.PIPELINE.standing_stats_bs,
                                                 latent_dim=gen.input[0].shape[1], num_classes=cfgs.DATA.num_classes, safe_copy=True, verbose=True)

        stddev_values = misc.get_list_or_tuple(cfgs.PIPELINE.stddev_search)

        for stddev in stddev_values:
            train_fns.run_pipeline_step(
                generator=gen,
                class_samples=class_samples,
                testing_dataset=testing_dataset,
                classifier_save_path=os.path.join(base_save_path, current_step, f'ckpt{str(ckpt_epoch).zfill(3)}'),
                cfgs=cfgs,
                classifier_pretrained=classifier_pretrained,
                filter=True,
                threshold=0.0,
                stddev=stddev,
                fixed_dset=False,
                recycling_period=10, # Fast Pipeline
                recycling_factor=1,
            )
            print("\n")


    ##################################################
    ###       STEP 3: Threshold optimization       ###
    ##################################################

    if 'all' in steps or 'threshold' in steps:
        current_step = 'threshold_optimization'
        misc.pretty_print(f"Running step: '{current_step}'", separator='#', spacing='\n\n')

        results_path = os.path.join(base_save_path, 'stddev_optimization', 'results.csv') # load results from previous step
        ckpt_results = pd.read_csv(results_path)
        best_ckpt = ckpt_results.iloc[ckpt_results['val_accuracy'].idxmax()]['checkpoint_name'] # get the best checkpoint from the previous step

        print("Best checkpoint is: ", best_ckpt)

        ckpt_epoch = int(re.search(r"ckpt(\d+)", best_ckpt).group(1))
        best_stddev = float(re.search(r"std(\d+(\.\d+)?)", best_ckpt).group(1))
        gen_path = os.path.join(base_load_path, f"*epoch*{str(ckpt_epoch).zfill(3)}*")
        gen = misc.load_model(model_path=gen_path, verbose=True)

        if cfgs.PIPELINE.apply_standing_stats:
            gen = misc.apply_standing_statistics(generator=gen, standing_max_batch=cfgs.PIPELINE.standing_stats_bs, standing_step=cfgs.PIPELINE.standing_stats_bs,
                                                 latent_dim=gen.input[0].shape[1], num_classes=cfgs.DATA.num_classes, safe_copy=True, verbose=True)

        threshold_values = misc.get_list_or_tuple(cfgs.PIPELINE.threshold_search)

        for threshold in threshold_values:
            train_fns.run_pipeline_step(
                generator=gen,
                class_samples=class_samples,
                testing_dataset=testing_dataset,
                classifier_save_path=os.path.join(base_save_path, current_step, f'ckpt{str(ckpt_epoch).zfill(3)}'),
                cfgs=cfgs,
                classifier_pretrained=classifier_pretrained,
                filter=True,
                threshold=threshold,
                stddev=best_stddev,
                fixed_dset=False,
                recycling_period=10, # Fast Pipeline
                recycling_factor=1,
            )
            print("\n")


    ##################################################
    ###        STEP 4: Best Hyperparameters        ###
    ##################################################

    if 'all' in steps or 'best' in steps:
        current_step = 'best_hyperparameters'
        misc.pretty_print(f"Running step: '{current_step}'", separator='#', spacing='\n\n')

        results_path = os.path.join(base_save_path, 'threshold_optimization', 'results.csv') # load results from previous step
        ckpt_results = pd.read_csv(results_path)
        best_ckpt = ckpt_results.iloc[ckpt_results['val_accuracy'].idxmax()]['checkpoint_name'] # get the best checkpoint from the previous step

        print("Best checkpoint is: ", best_ckpt)

        ckpt_epoch = int(re.search(r"ckpt(\d+)", best_ckpt).group(1))
        best_stddev = float(re.search(r"std(\d+(\.\d+)?)", best_ckpt).group(1))
        best_threshold = float(re.search(r"filter(\d+(\.\d+)?)", best_ckpt).group(1))
        gen_path = os.path.join(base_load_path, f"*epoch*{str(ckpt_epoch).zfill(3)}*")
        gen = misc.load_model(model_path=gen_path, verbose=True)

        if cfgs.PIPELINE.apply_standing_stats:
            gen = misc.apply_standing_statistics(generator=gen, standing_max_batch=cfgs.PIPELINE.standing_stats_bs, standing_step=cfgs.PIPELINE.standing_stats_bs,
                                                 latent_dim=gen.input[0].shape[1], num_classes=cfgs.DATA.num_classes, safe_copy=True, verbose=True)

        result = train_fns.run_pipeline_step(
            generator=gen,
            class_samples=class_samples,
            testing_dataset=testing_dataset,
            classifier_save_path=os.path.join(base_save_path, current_step, f'ckpt{str(ckpt_epoch).zfill(3)}'),
            cfgs=cfgs,
            classifier_pretrained=classifier_pretrained,
            filter=True,
            threshold=best_threshold,
            stddev=best_stddev,
            fixed_dset=False,
            recycling_period=1, # full recycle (i.e. Accurate Pipeline)
            recycling_factor=1,
        )
        print("\n")

        print(f"At the end of the pipeline, the {cfgs.PIPELINE.gan_name} model achieved an accuracy of: {max(result['history']['val_accuracy'])*100:.2f}%")

    ### End of pipeline ###
    misc.pretty_print(f"DONE", separator='#', spacing='\n\n')

def main():
    parser = parser_utils.prepare_pipeline_parser()
    args = parser.parse_args()
    args_dict = parser_utils.split_args(args)
    cfgs = config.PipelineConfiguration(args.cfg_file)
    cfgs.update_cfgs(args_dict)
    run(cfgs)

if __name__ == '__main__':
  main()