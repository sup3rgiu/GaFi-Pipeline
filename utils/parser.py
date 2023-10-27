import argparse
import ast

def prepare_classifier_parser():
    usage = 'Parser for train classifier script.\nAll arguments are optional, and if provided, they will override the default configuration or the configuration loaded from the configuration file.'
    parser = argparse.ArgumentParser(description=usage)

    ##########################################################################################################
    # Note that the 'default' values are all later replaced with argparse.SUPPRESS.                          #    
    # We keep the default values only for legacy/clarity, but the actual default values are in the config.py #
    ##########################################################################################################

    ### Config file ###
    parser.add_argument(
        '--cfg_file', type=str, default=None,
        help='If specified, overrides the default or given configuration with values in this file.')

    ### Dataset/Dataloader stuff ###
    parser.add_argument(
        '--dataset', type=str, default='CIFAR10', dest='DATA.dataset',
        help='Which Dataset to train on, out of FashionMNIST, CIFAR10, CIFAR100, CINIC10, DermaMNIST') 
    parser.add_argument(
        '--img_size', type=int, default=32, dest='DATA.image_size',
        help='Dimension to which the images should be resized. Resizing is only applied if the specified dimension is different from the original size of the dataset')
    parser.add_argument(
       '--categorical', type=str2bool, nargs='?', const=True, default=True, dest='DATA.categorical',
        help='Whether to convert the labels of the training set to categorical')
    parser.add_argument(
        '--val_categorical', type=str2bool, nargs='?', const=True, default=True, dest='DATA.val_categorical',
        help='Whether to convert the labels of the val/test sets to categorical')
    parser.add_argument(
        '--normalize', type=str2bool, nargs='?', const=True, default=True, dest='DATA.normalize',
        help='Whether to normalize the images')
    parser.add_argument(
        '--dequantize', type=str2bool, nargs='?', const=True, default=False, dest='DATA.dequantize',
        help='Whether to dequantize the images')
    parser.add_argument(
        '--resize', type=str2bool, nargs='?', const=True, default=None, dest='DATA.resize',
        help='Whether to resize the images to img_size')
    parser.add_argument(
        '--padding', type=str2bool, nargs='?', const=True, default=None, dest='DATA.padding',
        help='Whether to zero pad the images to img_size')
    parser.add_argument(
       '--horizontal_flip', type=str2bool, nargs='?', const=True, default=False, dest='DATA.horizontal_flip',
        help='Whether to augment with random horizontal flip at dataset level')
    parser.add_argument(
        '--merge_train_val', type=str2bool, nargs='?', const=True, default=False, dest='DATA.merge_train_val',
        help='Whether to merge train and val sets')
    parser.add_argument(
        '--drop_remainder', type=str2bool, nargs='?', const=True, default=False, dest='DATA.drop_remainder',
        help='Whether the last batch should be dropped in the case it has fewer samples than batch_size elements')

    ### Optimization stuff ###
    parser.add_argument(
        '--optimizer', type=str, default='sgdw', dest='OPTIMIZATION.optimizer',
        help='Which Optimizer to use, out of adam, sgd, sgdw, adabelief')
    parser.add_argument(
        '--lr', type=float, default=0.1, dest='OPTIMIZATION.lr',
        help='Learning rate to use')
    parser.add_argument(
        '--momentum', type=float, default=0.9, dest='OPTIMIZATION.momentum',
        help='Momentum to use for SGD/SGDW')
    parser.add_argument(
        '--nesterov', type=str2bool, nargs='?', const=True, default=True, dest='OPTIMIZATION.nesterov',
        help='Whether to apply Nesterov momentum for SGD/SGDW')
    parser.add_argument(
        '--weight_decay', type=float, default=1e-4, dest='OPTIMIZATION.weight_decay',
        help='Weight decay to use for SGDW')
    parser.add_argument(
        '--batch_size', type=int, default=128, dest='OPTIMIZATION.batch_size',
        help='Batch size for training')
    parser.add_argument(
        '--val_batch_size', type=int, default=128, dest='OPTIMIZATION.val_batch_size',
        help='Batch size for evaluation')
    parser.add_argument(
        '--epochs', type=int, default=400, dest='OPTIMIZATION.epochs',
        help='Training epochs. Only the best model is kept')

    ### Model stuff ###
    parser.add_argument(
        '--model', type=str, default='resnet20', dest='MODEL.name',
        help='Which Model to use, out of resnetXX, resnet_studiogan, simple')
    parser.add_argument(
        '--width', type=int, default=64, dest='MODEL.width',
        help='Model width')
    
    ### Augmentation stuff ###
    parser.add_argument(
       '--random_flip', type=str2bool, nargs='?', const=True, default=True, dest='AUG.random_flip',
        help='Whether to augment with random horizontal flip')
    parser.add_argument(
       '--random_crop', type=str2bool, nargs='?', const=True, default=True, dest='AUG.random_crop',
        help='Whether to augment with random crop')
    parser.add_argument(
       '--random_rotation', type=str2bool, nargs='?', const=True, default=False, dest='AUG.random_rotation',
        help='Whether to augment with random rotation')
    parser.add_argument(
       '--random_zoom', type=str2bool, nargs='?', const=True, default=False, dest='AUG.random_zoom',
        help='Whether to augment with random zoom')
    parser.add_argument(
       '--random_erasing', type=str2bool, nargs='?', const=True, default=True, dest='AUG.random_erasing',
       help='Whether to augment with random erasing')
    
    ### Run stuff ###
    parser.add_argument(
        '--seed', type=int, default=42, dest='RUN.seed',
        help='Random seed')
    parser.add_argument(
        '--save_path', type=str, default='./save/Models/Classifiers', dest='RUN.save_path',
        help='Base path where to save the model')
    parser.add_argument('--extra_name', type=str, default='', dest='RUN.extra_name',
        help='Extra name to append to the (automatically generated) model name')
    parser.add_argument(
       '--mixed_precision', type=str2bool, nargs='?', const=True, default=False, dest='RUN.mixed_precision',
       help='Whether to use mixed precision')
    
    suppress_default_values(parser, exclude=['cfg_file'])
    fix_metavar(parser)
    
    return parser

def prepare_gan_parser():
    usage = 'Parser for train gan script.\nAll arguments are optional, and if provided, they will override the default configuration or the configuration loaded from the configuration file.'
    parser = argparse.ArgumentParser(description=usage)

    ##########################################################################################################
    # Note that the 'default' values are all later replaced with argparse.SUPPRESS.                          #    
    # We keep the default values only for legacy/clarity, but the actual default values are in the config.py #
    ##########################################################################################################

    ### Config file ###
    parser.add_argument(
        '--cfg_file', type=str, default=None,
        help='If specified, overrides the default or given configuration with values in this file.')
    
    ### Dataset/Dataloader stuff ###
    parser.add_argument(
        '--dataset', type=str, default='CIFAR10', dest='DATA.dataset',
        help='Which Dataset to train on, out of FashionMNIST, CIFAR10, CIFAR100, CINIC10, DermaMNIST') 
    parser.add_argument(
        '--img_size', type=int, default=32, dest='DATA.image_size',
        help='Dimension to which the images should be resized. Resizing is only applied if the specified dimension is different from the original size of the dataset')
    parser.add_argument(
       '--categorical', type=str2bool, nargs='?', const=True, default=False, dest='DATA.categorical',
        help='Whether to convert the labels of the training set to categorical')
    parser.add_argument(
        '--val_categorical', type=str2bool, nargs='?', const=True, default=True, dest='DATA.val_categorical',
        help='Whether to convert the labels of the val/test sets to categorical')
    parser.add_argument(
        '--normalize', type=str2bool, nargs='?', const=True, default=True, dest='DATA.normalize',
        help='Whether to normalize the images')
    parser.add_argument(
        '--dequantize', type=str2bool, nargs='?', const=True, default=False, dest='DATA.dequantize',
        help='Whether to dequantize the images')
    parser.add_argument(
        '--resize', type=str2bool, nargs='?', const=True, default=None, dest='DATA.resize',
        help='Whether to resize the images to img_size')
    parser.add_argument(
        '--padding', type=str2bool, nargs='?', const=True, default=None, dest='DATA.padding',
        help='Whether to zero pad the images to img_size')
    parser.add_argument(
       '--horizontal_flip', type=str2bool, nargs='?', const=True, default=True, dest='DATA.horizontal_flip',
        help='Whether to augment with random horizontal flip at dataset level')
    parser.add_argument(
        '--merge_train_val', type=str2bool, nargs='?', const=True, default=True, dest='DATA.merge_train_val',
        help='Whether to merge train and val sets')
    parser.add_argument(
        '--drop_remainder', type=str2bool, nargs='?', const=True, default=False, dest='DATA.drop_remainder',
        help='Whether the last batch should be dropped in the case it has fewer samples than batch_size elements')

    ### Optimization stuff ###
    parser.add_argument('--optimizer', type=str, default='adam', dest='OPTIMIZATION.optimizer',
        help='Which Optimizer to use')
    parser.add_argument('--g_lr', type=float, default=2e-4, dest='OPTIMIZATION.g_lr',
        help='Learning rate for Generator optimizer')
    parser.add_argument('--g_beta1', type=float, default=0.5, dest='OPTIMIZATION.g_beta1',
        help='Beta1 value for Adam optimizer for Generator')
    parser.add_argument('--g_beta2', type=float, default=0.999, dest='OPTIMIZATION.g_beta2',
        help='Beta2 value for Adam optimizer for Generator')
    parser.add_argument('--d_lr', type=float, default=2e-4, dest='OPTIMIZATION.d_lr',
        help='Learning rate for Discriminator optimizer')
    parser.add_argument('--d_beta1', type=float, default=0.5, dest='OPTIMIZATION.d_beta1',
        help='Beta1 value for Adam optimizer for Discriminator')
    parser.add_argument('--d_beta2', type=float, default=0.999, dest='OPTIMIZATION.d_beta2',
        help='Beta2 value for Adam optimizer for Discriminator')
    parser.add_argument('--adam_eps', type=float, default=1e-6, dest='OPTIMIZATION.adam_eps',
        help='Adam constant for numerical stability')
    parser.add_argument('--g_updates_per_step', type=int, default=1, dest='OPTIMIZATION.g_updates_per_step',
        help='The number of generator updates per step')
    parser.add_argument('--d_updates_per_step', type=int, default=3, dest='OPTIMIZATION.d_updates_per_step',
        help='The number of discriminator updates per step')
    parser.add_argument('--split_batch_d_steps', type=str2bool, nargs='?', const=True, default=False, dest='OPTIMIZATION.split_batch_d_steps',
        help='Whether to use the traditional interpretation of d_updates_per_step or the one proposed in our work (True: traditional, False: ours)')
    parser.add_argument('--acml_steps', type=int, default=1, dest='OPTIMIZATION.acml_steps',
        help='Accumulation steps for large batch training')
    parser.add_argument('--random_labels', type=str2bool, nargs='?', const=True, default=False, dest='OPTIMIZATION.random_labels',
        help='Whether to use random labels during training. If False, the labels will be the ones of the current real batch')
    parser.add_argument('--batch_size', type=int, default=64, dest='OPTIMIZATION.batch_size',
        help='Batch size for training (actual batch size will be batch_size * d_updates_per_step * acml_steps)')
    parser.add_argument('--val_batch_size', type=int, default=128, dest='OPTIMIZATION.val_batch_size',
        help='Batch size for evaluation')
    parser.add_argument('--total_steps', type=int, default=100000, dest='OPTIMIZATION.total_steps',
        help='Training steps')
    parser.add_argument('--epochs', type=int, default=500, dest='OPTIMIZATION.epochs',
        help='Training epochs (None to use total_steps)')
    parser.add_argument('--save_optimizer', type=str2bool, nargs='?', const=True, default=True, dest='OPTIMIZATION.save_optimizer',
        help='Whether to save the optimizer state at the end of the training')
    parser.add_argument('--save_optimizer_ckpt', type=str2bool, nargs='?', const=True, default=False, dest='OPTIMIZATION.save_optimizer_ckpt',
        help='Whether to save the optimizer state for each model checkpoint')
    
    ### Model stuff ###
    parser.add_argument('--model_name', type=str, default='biggan_deep', dest='MODEL.name',
        help='Model name, out of biggan, biggan_deep')
    parser.add_argument('--studiogan', type=str2bool, nargs='?', const=True, default=True, dest='MODEL.studiogan',
        help='Whether to use StudioGAN version of G_Resblock_Deep and D_Resblock_Deep')
    parser.add_argument('--hier', type=str2bool, nargs='?', const=True, default=True, dest='MODEL.hier',
        help='Whether to use hierarchical noise z in generator')
    parser.add_argument('--latent_dim', type=int, default=128, dest='MODEL.latent_dim',
        help='Dimension of noise vectors')
    parser.add_argument('--shared_dim', type=int, default=128, dest='MODEL.shared_dim',
        help='Dimension of shared latent embedding')
    parser.add_argument('--g_conv_dim', type=int, default=128, dest='MODEL.g_conv_dim',
        help='Base channel (i.e. channel multiplier) for the resnet style generator architecture')
    parser.add_argument('--g_depth', type=int, default=2, dest='MODEL.g_depth',
        help='Generator depth for biggan_deep')
    parser.add_argument('--apply_g_attn', type=str2bool, nargs='?', const=True, default=True, dest='MODEL.apply_g_attn',
        help='Whether to apply self-attention proposed by zhang et al. (SAGAN) inside generator architecture')
    parser.add_argument('--d_conv_dim', type=int, default=128, dest='MODEL.d_conv_dim',
        help='Base channel (i.e. channel multiplier) for the resnet style discriminator architecture')
    parser.add_argument('--d_depth', type=int, default=2, dest='MODEL.d_depth',
        help='Discriminator depth for biggan_deep')
    parser.add_argument('--apply_d_attn', type=str2bool, nargs='?', const=True, default=True, dest='MODEL.apply_d_attn',
        help='Whether to apply self-attention proposed by zhang et al. (SAGAN) inside discriminator architecture')
    parser.add_argument('--d_wide', type=str2bool, nargs='?', const=True, default=True, dest='MODEL.d_wide',
        help='Whether to use the SN-GAN channel pattern for discriminator')
    parser.add_argument('--blur_resample', type=str2bool, nargs='?', const=True, default=False, dest='MODEL.blur_resample',
        help='Whether to use blur resample instead of average pooling for discriminator downsample')
    parser.add_argument('--residual_concat', type=str2bool, nargs='?', const=True, default=False, dest='MODEL.residual_concat',
        help='Whether to use Concationation+Conv1x1 instead of Sum for residual skip connections')
    parser.add_argument('--apply_g_ema', type=str2bool, nargs='?', const=True, default=True, dest='MODEL.apply_g_ema',
        help='Whether to apply moving average update for the generator')
    parser.add_argument('--g_ema_decay', type=float, default=0.999, dest='MODEL.g_ema_decay',
        help='Decay rate for the ema generator')
    parser.add_argument('--g_ema_start', type=int, default=1000, dest='MODEL.g_ema_start',
        help='Starting step for g_ema update')
    
    ### Loss stuff ###
    parser.add_argument('--loss_type', type=str, default='hinge', dest='LOSS.type',
        help='Loss name, out of hinge, non-saturating')
    parser.add_argument('--grad_penalty_type', type=str, default=None, dest='LOSS.grad_penalty_type',
        help='Gradient pentalty type, out of r1, None')
    parser.add_argument('--grad_penalty_cost', type=float, default=10.0, dest='LOSS.grad_penalty_cost',
        help='Gradient penalty weight')
    
    ### Classifier pretrained stuff (used to automatically determine the model name) ###
    parser.add_argument('--cls_pre_name', type=str, default='resnet20', dest='CLS_PRE.name',
        help='Name of the pretrained classifier architecture') 
    parser.add_argument('--cls_pre_optimizer', type=str, default='sgdw', dest='CLS_PRE.optimizer',
        help='Optimizer type of the pretrained classifier')
    parser.add_argument('--cls_pre_random_erasing', type=str2bool, nargs='?', const=True, default=True, dest='CLS_PRE.random_erasing',
        help='Whether the random erasing has been used for the pretrained classifier')
    parser.add_argument('--cls_pre_extra_name', type=str, default='', dest='CLS_PRE.extra_name',
        help='Extra name to append to the (automatically generated) model name of the pretrained classifier')
    parser.add_argument('--cls_pre_save_path', type=str, default='./save/Models/Classifiers', dest='CLS_PRE.save_path',
        help='Base path where the pretrained classifier is saved')

    ### Run stuff ###
    parser.add_argument('--seed', type=int, default=42, dest='RUN.seed',
        help='Random seed')
    parser.add_argument('--save', type=str2bool, nargs='?', const=True, default=True, dest='RUN.save',
        help='Whether to save anything produced by the run (i.e. model, images, history, etc.)')
    parser.add_argument('--save_path', type=str, default='./save/Models/BigGAN', dest='RUN.save_path',
        help='Base path where to save the model')
    parser.add_argument('--extra_name', type=str, default='', dest='RUN.extra_name',
        help='Extra name to append to the (automatically generated) model name')
    parser.add_argument('--mixed_precision', type=str2bool, nargs='?', const=True, default=False, dest='RUN.mixed_precision',
        help='Whether to use mixed precision')
    parser.add_argument('--compute_metrics', type=str2bool, nargs='?', const=True, default=False, dest='RUN.compute_metrics',
        help='Whether to compute metrics at the end of each epoch. If True, the training will be much slower')
    parser.add_argument('--start_eval_epoch', type=int, default=0, dest='RUN.start_eval_epoch',
        help='Epoch at which start computing evaluation metrics')
    parser.add_argument('--compute_classifier_stats', type=str2bool, nargs='?', const=True, default=False, dest='RUN.compute_classifier_stats',
        help='Whether to compute stats associated to the pretrained classifier during GAN training')
    parser.add_argument('--reload', type=str2bool, nargs='?', const=True, default=False, dest='RUN.reload',
        help='Whether to reload the model (gen, disc, history, etc.) if it already exists (useful for resuming training)')
    parser.add_argument('--show_plots', type=str2bool, nargs='?', const=True, default=False, dest='RUN.show_plots',
        help='Whether to plot the images during training (useful if running in a Jupyter notebook)')
    parser.add_argument('--plot_with_title', type=str2bool, nargs='?', const=True, default=False, dest='RUN.plot_with_title',
        help='Whether to add a title to the plot of the generated images')
    parser.add_argument('--plot_title', type=str, default='', dest='RUN.plot_title',
        help='Title to add to the plot of the generated images')
    parser.add_argument('--keep_images', type=str2bool, nargs='?', const=True, default=False, dest='RUN.keep_images',
        help='Whether to keep the single generated images. If False, only the GIF will be kept')
    parser.add_argument('--fixed_seed', type=str2bool, nargs='?', const=True, default=True, dest='RUN.fixed_seed',
        help='Whether to fix the seed at the beginning of each epoch (theoretically, should be False)')
    
    suppress_default_values(parser, exclude=['cfg_file'])
    fix_metavar(parser)

    return parser

def prepare_pipeline_parser():
    usage = 'Parser to run the pipeline script.\nAll arguments are optional, and if provided, they will override the default configuration or the configuration loaded from the configuration file.'
    parser = argparse.ArgumentParser(description=usage)

    ##########################################################################################################
    # Note that the 'default' values are all later replaced with argparse.SUPPRESS.                          #    
    # We keep the default values only for legacy/clarity, but the actual default values are in the config.py #
    ##########################################################################################################

    ### Config file ###
    parser.add_argument(
        '--cfg_file', type=str, default=None,
        help='If specified, overrides the default or given configuration with values in this file.')

    ### Dataset/Dataloader stuff ###
    parser.add_argument(
        '--dataset', type=str, default='CIFAR10', dest='DATA.dataset',
        help='Which Dataset to train on, out of FashionMNIST, CIFAR10, CIFAR100, CINIC10, DermaMNIST') 
    parser.add_argument(
        '--img_size', type=int, default=32, dest='DATA.image_size',
        help='Dimension to which the images should be resized. Resizing is only applied if the specified dimension is different from the original size of the dataset')
    parser.add_argument(
       '--categorical', type=str2bool, nargs='?', const=True, default=True, dest='DATA.categorical',
        help='Whether to convert the labels of the training set to categorical')
    parser.add_argument(
        '--val_categorical', type=str2bool, nargs='?', const=True, default=True, dest='DATA.val_categorical',
        help='Whether to convert the labels of the val/test sets to categorical')
    parser.add_argument(
        '--normalize', type=str2bool, nargs='?', const=True, default=True, dest='DATA.normalize',
        help='Whether to normalize the images')
    parser.add_argument(
        '--dequantize', type=str2bool, nargs='?', const=True, default=False, dest='DATA.dequantize',
        help='Whether to dequantize the images')
    parser.add_argument(
        '--resize', type=str2bool, nargs='?', const=True, default=None, dest='DATA.resize',
        help='Whether to resize the images to img_size')
    parser.add_argument(
        '--padding', type=str2bool, nargs='?', const=True, default=None, dest='DATA.padding',
        help='Whether to zero pad the images to img_size')
    parser.add_argument(
       '--horizontal_flip', type=str2bool, nargs='?', const=True, default=False, dest='DATA.horizontal_flip',
        help='Whether to augment with random horizontal flip at dataset level')
    parser.add_argument(
        '--merge_train_val', type=str2bool, nargs='?', const=True, default=False, dest='DATA.merge_train_val',
        help='Whether to merge train and val sets')
    parser.add_argument(
        '--drop_remainder', type=str2bool, nargs='?', const=True, default=False, dest='DATA.drop_remainder',
        help='Whether the last batch should be dropped in the case it has fewer samples than batch_size elements')

    ### Optimization stuff ###
    parser.add_argument(
        '--optimizer', type=str, default='sgdw', dest='OPTIMIZATION.optimizer',
        help='Which Optimizer to use, out of adam, sgd, sgdw, adabelief')
    parser.add_argument(
        '--lr', type=float, default=0.1, dest='OPTIMIZATION.lr',
        help='Learning rate to use')
    parser.add_argument(
        '--momentum', type=float, default=0.9, dest='OPTIMIZATION.momentum',
        help='Momentum to use for SGD/SGDW')
    parser.add_argument(
        '--nesterov', type=str2bool, nargs='?', const=True, default=True, dest='OPTIMIZATION.nesterov',
        help='Whether to apply Nesterov momentum for SGD/SGDW')
    parser.add_argument(
        '--weight_decay', type=float, default=1e-4, dest='OPTIMIZATION.weight_decay',
        help='Weight decay to use for SGDW')
    parser.add_argument(
        '--batch_size', type=int, default=128, dest='OPTIMIZATION.batch_size',
        help='Batch size for training')
    parser.add_argument(
        '--val_batch_size', type=int, default=128, dest='OPTIMIZATION.val_batch_size',
        help='Batch size for evaluation')
    parser.add_argument(
        '--epochs', type=int, default=100, dest='OPTIMIZATION.epochs',
        help='Training epochs. Only the best model is kept')

    ### Model stuff ###
    parser.add_argument(
        '--model', type=str, default='resnet20', dest='MODEL.name',
        help='Which Model to use, out of resnetXX, resnet_studiogan, simple')
    parser.add_argument(
        '--width', type=int, default=64, dest='MODEL.width',
        help='Model width')
    
    ### Pipeline stuff ###
    parser.add_argument(
        '--gan_name', type=str, default=None, dest='PIPELINE.gan_name',
        help='Name of the GAN run to use in the pipeline. Must be specified manually')
    parser.add_argument(
        '--steps', type=lambda s: s.split(','), default='all', dest='PIPELINE.steps',
        help='Comma-seperated string containing the desired pipeline steps in ["all", "ckpt", "stddev", "threshold", "best"]')
    parser.add_argument(
        '--ckpt_epochs', type=lambda s: ast.literal_eval(s), default=(150, 500, 10), dest='PIPELINE.ckpt_epochs',
        help='Which checkpoints evaluate in the "checkpoint optimization" step of the pipeline. A list of integers to evaluate the corresponding checkpoints, or a triplet (start, end, step) provided as string to evaluate all the checkpoints in the range [start, end] with the given step')
    parser.add_argument(
        '--stddev_search', type=lambda s: ast.literal_eval(s), default=(1, 2, 0.05), dest='PIPELINE.stddev_search',
        help='Which values of standard deviation evaluate in the "stddev optimization" step of the pipeline. A list of floats to evaluate the corresponding standard deviations, or a triplet (start, end, step) provided as string to evaluate all the standard deviations in the range [start, end] with the given step')
    parser.add_argument(
        '--threshold_search', type=lambda s: ast.literal_eval(s), default=(0, 0.9, 0.1), dest='PIPELINE.threshold_search',
        help='Which values of threshold evaluate in the "threshold optimization" step of the pipeline. A list of floats to evaluate the corresponding thresholds, or a triplet (start, end, step) provided as string to evaluate all the thresholds in the range [start, end] with the given step')
    parser.add_argument(
        '--load_path', type=str, default='./save/Models/BigGAN', dest='PIPELINE.load_path',
        help='Base path where GAN model is saved')
    parser.add_argument(
        '--apply_standing_stats', type=str2bool, nargs='?', const=True, default=False, dest='PIPELINE.apply_standing_stats',
        help='Whether to apply the standing stats trick on the generator before its first use')
    parser.add_argument(
        '--standing_stats_bs', type=int, default=192, dest='PIPELINE.standing_stats_bs',
        help='Batch size to use when applying the standing stats trick (the same batch size used during generator training works well)')
    parser.add_argument(
        '--filtering_attempts', type=int, default=-1, dest='PIPELINE.filtering_attempts',
        help='How many times to try to filter the dataset before adding unfiltered samples (if -1, there is no attempts limit). Useful for datasets with many classes, where the generator could be collapsed for one or more classes')
    parser.add_argument(
        '--class_samples', type=int, default=None, dest='PIPELINE.class_samples',
        help='Number of samples per class to generate for the fake dataset. If None, the number of samples per class will be computed to have the same number of samples as the real dataset')
    
    ### Augmentation stuff ###
    parser.add_argument(
       '--random_flip', type=str2bool, nargs='?', const=True, default=True, dest='AUG.random_flip',
        help='Whether to augment with random horizontal flip')
    parser.add_argument(
       '--random_crop', type=str2bool, nargs='?', const=True, default=True, dest='AUG.random_crop',
        help='Whether to augment with random crop')
    parser.add_argument(
       '--random_rotation', type=str2bool, nargs='?', const=True, default=False, dest='AUG.random_rotation',
        help='Whether to augment with random rotation')
    parser.add_argument(
       '--random_zoom', type=str2bool, nargs='?', const=True, default=False, dest='AUG.random_zoom',
        help='Whether to augment with random zoom')
    parser.add_argument(
       '--random_erasing', type=str2bool, nargs='?', const=True, default=True, dest='AUG.random_erasing',
       help='Whether to augment with random erasing')
    
    ### Classifier pretrained stuff (used to automatically determine the model name) ###
    parser.add_argument('--cls_pre_name', type=str, default='resnet20', dest='CLS_PRE.name',
        help='Name of the pretrained classifier architecture') 
    parser.add_argument('--cls_pre_optimizer', type=str, default='sgdw', dest='CLS_PRE.optimizer',
        help='Optimizer type of the pretrained classifier')
    parser.add_argument('--cls_pre_random_erasing', type=str2bool, nargs='?', const=True, default=True, dest='CLS_PRE.random_erasing',
        help='Whether the random erasing has been used for the pretrained classifier')
    parser.add_argument('--cls_pre_extra_name', type=str, default='', dest='CLS_PRE.extra_name',
        help='Extra name to append to the (automatically generated) model name of the pretrained classifier')
    parser.add_argument('--cls_pre_save_path', type=str, default='./save/Models/Classifiers', dest='CLS_PRE.save_path',
        help='Base path where the pretrained classifier is saved')
    
    ### Run stuff ###
    parser.add_argument(
        '--seed', type=int, default=42, dest='RUN.seed',
        help='Random seed')
    parser.add_argument(
        '--save_path', type=str, default='./save/Models/Classifiers', dest='RUN.save_path',
        help='Base path where to save the model')
    parser.add_argument('--extra_name', type=str, default='', dest='RUN.extra_name',
        help='Extra name to append to the (automatically generated) model name')
    parser.add_argument(
       '--mixed_precision', type=str2bool, nargs='?', const=True, default=False, dest='RUN.mixed_precision',
       help='Whether to use mixed precision')
    
    suppress_default_values(parser, exclude=['cfg_file'])
    fix_metavar(parser)
    
    return parser

def prepare_multigan_parser():
    usage = 'Parser to run the MultiGAN pipeline script.\nAll arguments are optional, and if provided, they will override the default configuration or the configuration loaded from the configuration file.'
    parser = argparse.ArgumentParser(description=usage)

    ##########################################################################################################
    # Note that the 'default' values are all later replaced with argparse.SUPPRESS.                          #    
    # We keep the default values only for legacy/clarity, but the actual default values are in the config.py #
    ##########################################################################################################

    ### Config file ###
    parser.add_argument(
        '--cfg_file', type=str, default=None,
        help='If specified, overrides the default or given configuration with values in this file.')

    ### Dataset/Dataloader stuff ###
    parser.add_argument(
        '--dataset', type=str, default='CIFAR10', dest='DATA.dataset',
        help='Which Dataset to train on, out of FashionMNIST, CIFAR10, CIFAR100, CINIC10, DermaMNIST') 
    parser.add_argument(
        '--img_size', type=int, default=32, dest='DATA.image_size',
        help='Dimension to which the images should be resized. Resizing is only applied if the specified dimension is different from the original size of the dataset')
    parser.add_argument(
       '--categorical', type=str2bool, nargs='?', const=True, default=True, dest='DATA.categorical',
        help='Whether to convert the labels of the training set to categorical')
    parser.add_argument(
        '--val_categorical', type=str2bool, nargs='?', const=True, default=True, dest='DATA.val_categorical',
        help='Whether to convert the labels of the val/test sets to categorical')
    parser.add_argument(
        '--normalize', type=str2bool, nargs='?', const=True, default=True, dest='DATA.normalize',
        help='Whether to normalize the images')
    parser.add_argument(
        '--dequantize', type=str2bool, nargs='?', const=True, default=False, dest='DATA.dequantize',
        help='Whether to dequantize the images')
    parser.add_argument(
        '--resize', type=str2bool, nargs='?', const=True, default=None, dest='DATA.resize',
        help='Whether to resize the images to img_size')
    parser.add_argument(
        '--padding', type=str2bool, nargs='?', const=True, default=None, dest='DATA.padding',
        help='Whether to zero pad the images to img_size')
    parser.add_argument(
       '--horizontal_flip', type=str2bool, nargs='?', const=True, default=False, dest='DATA.horizontal_flip',
        help='Whether to augment with random horizontal flip at dataset level')
    parser.add_argument(
        '--merge_train_val', type=str2bool, nargs='?', const=True, default=False, dest='DATA.merge_train_val',
        help='Whether to merge train and val sets')
    parser.add_argument(
        '--drop_remainder', type=str2bool, nargs='?', const=True, default=False, dest='DATA.drop_remainder',
        help='Whether the last batch should be dropped in the case it has fewer samples than batch_size elements')

    ### Optimization stuff ###
    parser.add_argument(
        '--optimizer', type=str, default='sgdw', dest='OPTIMIZATION.optimizer',
        help='Which Optimizer to use, out of adam, sgd, sgdw, adabelief')
    parser.add_argument(
        '--lr', type=float, default=0.1, dest='OPTIMIZATION.lr',
        help='Learning rate to use')
    parser.add_argument(
        '--momentum', type=float, default=0.9, dest='OPTIMIZATION.momentum',
        help='Momentum to use for SGD/SGDW')
    parser.add_argument(
        '--nesterov', type=str2bool, nargs='?', const=True, default=True, dest='OPTIMIZATION.nesterov',
        help='Whether to apply Nesterov momentum for SGD/SGDW')
    parser.add_argument(
        '--weight_decay', type=float, default=1e-4, dest='OPTIMIZATION.weight_decay',
        help='Weight decay to use for SGDW')
    parser.add_argument(
        '--batch_size', type=int, default=128, dest='OPTIMIZATION.batch_size',
        help='Batch size for training')
    parser.add_argument(
        '--val_batch_size', type=int, default=128, dest='OPTIMIZATION.val_batch_size',
        help='Batch size for evaluation')
    parser.add_argument(
        '--epochs', type=int, default=100, dest='OPTIMIZATION.epochs',
        help='Training epochs. Only the best model is kept')

    ### Model stuff ###
    parser.add_argument(
        '--model', type=str, default='resnet20', dest='MODEL.name',
        help='Which Model to use, out of resnetXX, resnet_studiogan, simple')
    parser.add_argument(
        '--width', type=int, default=64, dest='MODEL.width',
        help='Model width')
    
    ### Pipeline stuff ###
    parser.add_argument(
        '--gan_names', type=lambda s: s.split(','), default=None, dest='PIPELINE.gan_names',
        help='Comma-seperated string containing the names of the GAN runs to use in the pipeline. Must be specified manually')
    parser.add_argument(
        '--load_path', type=str, default='./save/Models/BigGAN', dest='PIPELINE.load_path',
        help='Base path where GAN model is saved')
    parser.add_argument(
        '--one_gan_for_epoch', type=str2bool, nargs='?', const=True, default=False, dest='PIPELINE.one_gan_for_epoch',
        help='Whether to use only one GAN at each trainig epoch (in a circular way), or all GANs at each epoch')
    parser.add_argument(
        '--apply_standing_stats', type=str2bool, nargs='?', const=True, default=False, dest='PIPELINE.apply_standing_stats',
        help='Whether to apply the standing stats trick on the generator before its first use')
    parser.add_argument(
        '--standing_stats_bs', type=int, default=192, dest='PIPELINE.standing_stats_bs',
        help='Batch size to use when applying the standing stats trick (the same batch size used duing generator training works well)')
    parser.add_argument(
        '--filtering_attempts', type=int, default=-1, dest='PIPELINE.filtering_attempts',
        help='How many times to try to filter the dataset before adding unfiltered samples (if filtering_attempts=-1, there is no attemps limit). Useful for datasets with many classes, where the generator could be collapsed for one or more classes')
    parser.add_argument(
        '--class_samples', type=list, default=None, dest='PIPELINE.class_samples',
        help='Number of samples per class to generate for the fake dataset. If None, the number of samples per class will be computed to have the same number of samples as the real dataset')
    parser.add_argument(
        '--best_class_samples', type=list, default=None, dest='PIPELINE.best_class_samples',
        help='Number of samples per class used during the pipeline steps to generate the fake datasets. If None, the number of samples per class will be computed to have the same number of samples as the real dataset. Useful to load the best hyperparameters from a pipeline trained with more/less samples than the one that will be used for the current MultiGAN pipeline')
    parser.add_argument(
        '--best_extra_name', type=str, default='', dest='PIPELINE.best_extra_name',
        help='Extra name to append to the (automatically generated) classifier name used in the "best hyperparameters" step')
    
    ### Augmentation stuff ###
    parser.add_argument(
       '--random_flip', type=str2bool, nargs='?', const=True, default=True, dest='AUG.random_flip',
        help='Whether to augment with random horizontal flip')
    parser.add_argument(
       '--random_crop', type=str2bool, nargs='?', const=True, default=True, dest='AUG.random_crop',
        help='Whether to augment with random crop')
    parser.add_argument(
       '--random_rotation', type=str2bool, nargs='?', const=True, default=False, dest='AUG.random_rotation',
        help='Whether to augment with random rotation')
    parser.add_argument(
       '--random_zoom', type=str2bool, nargs='?', const=True, default=False, dest='AUG.random_zoom',
        help='Whether to augment with random zoom')
    parser.add_argument(
       '--random_erasing', type=str2bool, nargs='?', const=True, default=True, dest='AUG.random_erasing',
       help='Whether to augment with random erasing')
       
    ### Classifier pretrained stuff (used to automatically determine the model name) ###
    parser.add_argument('--cls_pre_name', type=str, default='resnet20', dest='CLS_PRE.name',
        help='Name of the pretrained classifier architecture') 
    parser.add_argument('--cls_pre_optimizer', type=str, default='sgdw', dest='CLS_PRE.optimizer',
        help='Optimizer type of the pretrained classifier')
    parser.add_argument('--cls_pre_random_erasing', type=str2bool, nargs='?', const=True, default=True, dest='CLS_PRE.random_erasing',
        help='Whether the random erasing has been used for the pretrained classifier')
    parser.add_argument('--cls_pre_extra_name', type=str, default='', dest='CLS_PRE.extra_name',
        help='Extra name to append to the (automatically generated) model name of the pretrained classifier')
    parser.add_argument('--cls_pre_save_path', type=str, default='./save/Models/Classifiers', dest='CLS_PRE.save_path',
        help='Base path where the pretrained classifier is saved')

    ### Run stuff ###
    parser.add_argument(
        '--seed', type=int, default=42, dest='RUN.seed',
        help='Random seed')
    parser.add_argument(
        '--save_path', type=str, default='./save/Models/Classifiers', dest='RUN.save_path',
        help='Base path where to save the model')
    parser.add_argument('--extra_name', type=str, default='', dest='RUN.extra_name',
        help='Extra name to append to the (automatically generated) model name')
    parser.add_argument(
       '--mixed_precision', type=str2bool, nargs='?', const=True, default=False, dest='RUN.mixed_precision',
       help='Whether to use mixed precision')
    
    suppress_default_values(parser, exclude=['cfg_file'])
    fix_metavar(parser)
    
    return parser

# If the 'dest' argument is supplied, then the metavar is set to it and is shown in the help message
# Instead I want to show the argument name as usual, or something custom for certain argument types
def fix_metavar(parser):
    for action in parser._actions:
        if action.type == str2bool:
            action.metavar = 'TRUE/FALSE'
        else:
            action.metavar = action.option_strings[-1].replace('-', '').upper()

# Replace all the default values (except for excluded arguments) with argparse.SUPPRESS
# In this way, if the user does not use the argument, it will not appear in the args namespace
def suppress_default_values(parser, exclude=[]):
    for action in parser._actions:
        if not isinstance(action, argparse._HelpAction) and action.dest not in exclude:
            action.default = argparse.SUPPRESS

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Split the keys by '.' and create nested dictionaries
def split_args(args):
    args_dict = {}
    for key, value in vars(args).items():
        keys = key.split('.')
        current_dict = args_dict
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                current_dict[k] = value
            else:
                if k not in current_dict:
                    current_dict[k] = {}
                current_dict = current_dict[k]
    return args_dict