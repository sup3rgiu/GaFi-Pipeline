import types
import yaml
import os
from json import JSONEncoder

# Style borrowed from https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/config.py

class ConfigEncoder(JSONEncoder):
    def default(self, obj):
        return obj.__dict__  

class Configuration(object):
    def __init__(self, cfg_file=None):
        self.cfg_file = cfg_file
        self.load_base_cfgs()
        self._overwrite_cfgs(self.cfg_file)

    def load_base_cfgs(self):
        raise NotImplementedError

    def _overwrite_cfgs(self, cfg_file):
        if cfg_file is not None and cfg_file != '' and os.path.exists(cfg_file):
            print("Loading custom configuration from {}".format(cfg_file))
            with open(cfg_file, 'r') as f:
                yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
                for super_cfg_name, attr_value in yaml_cfg.items():
                    for attr, value in attr_value.items():
                        # TODO: Currently yaml file only supports 2-levels of nesting, and exactly 2
                        if hasattr(self, super_cfg_name) and hasattr(getattr(self, super_cfg_name), attr):
                            setattr(getattr(self, super_cfg_name), attr, value)
                        else:
                            raise AttributeError("There does not exist '{cls}.{attr}' attribute in the config.py.". \
                                                format(cls=super_cfg_name, attr=attr))
                        
    def update_cfgs(self, dictionary, namespace=None):
        if namespace is None: namespace = self
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # If the value is a dictionary, create a new empty Namespace if needed and set its attributes recursively
                if not hasattr(namespace, key):
                    setattr(namespace, key, types.SimpleNamespace())
                self.update_cfgs(value, getattr(namespace, key))
            else:
                # Otherwise, set the attribute directly
                setattr(namespace, key, value)  


class ClassifierConfiguration(Configuration):
    def __init__(self, cfg_file=None):
        super().__init__(cfg_file)

    def load_base_cfgs(self):
        print("Loading base configuration for Classifier training")
        # -----------------------------------------------------------------------------
        # Data settings
        # -----------------------------------------------------------------------------
        self.DATA = types.SimpleNamespace()

        # dataset name \in ["FashionMNIST", "CIFAR10", "CIFAR100", "CINIC10", "DermaMNIST"]
        self.DATA.dataset = "CIFAR10"
        # image size for training
        self.DATA.image_size = 32
        # whether to convert the labels of the training set to categorical
        self.DATA.categorical = True
        # whether to convert the labels of the val/test sets to categorical
        self.DATA.val_categorical = True
        # whether to normalize the images
        self.DATA.normalize = True
        # whether to dequantize the images
        self.DATA.dequantize = False
        # whether to resize the images to img_size
        self.DATA.resize = None
        # wheter to zero pad the images to img_size
        self.DATA.padding = None
        # whether to apply random horizontal flip at dataset level
        self.DATA.horizontal_flip = False
        # whether to merge train and val sets
        self.DATA.merge_train_val = False
        # whether the last batch should be dropped in the case it has fewer samples than batch_size elements
        self.DATA.drop_remainder = False

        # -----------------------------------------------------------------------------
        # Optimizer settings
        # -----------------------------------------------------------------------------
        self.OPTIMIZATION = types.SimpleNamespace()

        # type of the optimizer \in ["adam", "sgd", "sgdw", "adabelief"]
        self.OPTIMIZATION.optimizer = "sgdw"
        # learning rate
        self.OPTIMIZATION.lr = 0.1
        # ,omentum to use for SGD/SGDW
        self.OPTIMIZATION.momentum = 0.9
        # wheter to apply Nesterov momentum for SGD/SGDW
        self.OPTIMIZATION.nesterov = True
        # weight decay to use for SGDW
        self.OPTIMIZATION.weight_decay = 1e-4
        # batch size for training
        self.OPTIMIZATION.batch_size = 128
        # batch size for evaluation
        self.OPTIMIZATION.val_batch_size = 128
        # training epochs
        self.OPTIMIZATION.epochs = 400

        # -----------------------------------------------------------------------------
        # Model settings
        # -----------------------------------------------------------------------------
        self.MODEL = types.SimpleNamespace()

        # model name \in ["resnetXX", "resnet_studiogan", "simple"]
        self.MODEL.name = "resnet20"
        # model width
        self.MODEL.width = 64

        # -----------------------------------------------------------------------------
        # Augmentation settings
        # -----------------------------------------------------------------------------
        self.AUG = types.SimpleNamespace()

        # where to apply random horizontal flip
        self.AUG.random_flip = True
        # where to apply random crop
        self.AUG.random_crop = True
        # where to apply random rotation
        self.AUG.random_rotation = False
        # where to apply random zoom
        self.AUG.random_zoom = False
        # where to apply random erasing
        self.AUG.random_erasing = True

        # -----------------------------------------------------------------------------
        # Run settings
        # -----------------------------------------------------------------------------
        self.RUN = types.SimpleNamespace()

        # random seed
        self.RUN.seed = 42
        # base path where to save the model
        self.RUN.save_path = './save/Models/Classifiers'
        # extra name to append to the (automatically generated) model name
        self.RUN.extra_name = ''
        # whether to use mixed precision
        self.RUN.mixed_precision = False


class GANConfiguration(Configuration):
    def __init__(self, cfg_file=None):
        super().__init__(cfg_file)

    def load_base_cfgs(self):
        print("Loading base configuration for GAN training")
        # -----------------------------------------------------------------------------
        # Data settings
        # -----------------------------------------------------------------------------
        self.DATA = types.SimpleNamespace()

        # dataset name \in ["FashionMNIST", "CIFAR10", "CIFAR100", "CINIC10", "DermaMNIST"]
        self.DATA.dataset = "CIFAR10"
        # image size for training
        self.DATA.image_size = 32
        # whether to convert the labels of the training set to categorical
        self.DATA.categorical = False
        # whether to convert the labels of the val/test sets to categorical
        self.DATA.val_categorical = True
        # whether to normalize the images
        self.DATA.normalize = True
        # whether to dequantize the images
        self.DATA.dequantize = False
        # whether to resize the images to img_size
        self.DATA.resize = None
        # wheter to zero pad the images to img_size
        self.DATA.padding = None
        # whether to apply random horizontal flip at dataset level
        self.DATA.horizontal_flip = True
        # whether to merge train and val sets
        self.DATA.merge_train_val = True
        # whether the last batch should be dropped in the case it has fewer samples than batch_size elements
        self.DATA.drop_remainder = True

        # -----------------------------------------------------------------------------
        # Optimizer settings
        # -----------------------------------------------------------------------------
        self.OPTIMIZATION = types.SimpleNamespace()

        # type of the optimizer \in ["adam"]
        self.OPTIMIZATION.optimizer = "adam"
        # learning rate for Generator optimizer
        self.OPTIMIZATION.g_lr = 2e-4
        # beta values for Adam optimizer for Generator
        self.OPTIMIZATION.g_beta1 = 0.5
        self.OPTIMIZATION.g_beta2 = 0.999
         # learning rate for Discriminator optimizer
        self.OPTIMIZATION.d_lr = 2e-4
        # beta values for Adam optimizer for Discriminator
        self.OPTIMIZATION.d_beta1 = 0.5
        self.OPTIMIZATION.d_beta2 = 0.999
        # Adam constant for numerical stability
        self.OPTIMIZATION.adam_eps = 1e-6
        # the number of generator updates per step
        # TODO: not implemented yet (to implement?)
        self.OPTIMIZATION.g_updates_per_step = 1
        # the number of discriminator updates per step
        self.OPTIMIZATION.d_updates_per_step = 3
        # whether to use the traditional interpretation of 'd_updates_per_step' or the one proposed in our work (True: traditional, False: ours)
        self.OPTIMIZATION.split_batch_d_steps = False
        # accumulation steps for large batch training
        # TODO: not implemented yet
        self.OPTIMIZATION.acml_steps = 1
        # whether to use random labels during training. If False, the labels will be the ones of the current real batch
        self.OPTIMIZATION.random_labels = False
        # batch size for training (actual batch size will be batch_size * d_updates_per_step * acml_steps)
        self.OPTIMIZATION.batch_size = 64
        # batch size for evaluation
        self.OPTIMIZATION.val_batch_size = 128
        # training steps
        self.OPTIMIZATION.total_steps = 100000
        # training epochs (None to use total_steps)
        self.OPTIMIZATION.epochs = 500
        # whether to save the optimizer state at the end of the training
        self.OPTIMIZATION.save_optimizer = True
        # whether to save the optimizer state for each model checkpoint
        self.OPTIMIZATION.save_optimizer_ckpt = False

        # -----------------------------------------------------------------------------
        # Model settings
        # -----------------------------------------------------------------------------
        self.MODEL = types.SimpleNamespace()

        # model name \in ["biggan", "biggan_deep"]
        self.MODEL.name = "biggan_deep"
        # whether to use StudioGAN version of G_Resblock_Deep and D_Resblock_Deep
        self.MODEL.studiogan = True
        # whether to use hierarchical noise z in generator
        self.MODEL.hier = True
        # dimension of noise vectors
        self.MODEL.latent_dim = 128
        # dimension of shared latent embedding
        self.MODEL.shared_dim = 128
        # base channel (i.e. channel multiplier) for the resnet style generator architecture
        self.MODEL.g_conv_dim = 128
        # generator depth for biggan_deep
        self.MODEL.g_depth = 2
        # whether to apply self-attention proposed by zhang et al. (SAGAN) inside generator architecture
        self.MODEL.apply_g_attn = True
        # base channel (i.e. channel multiplier) for the resnet style discriminator architecture
        self.MODEL.d_conv_dim = 128
        # discriminator depth for biggan_deep
        self.MODEL.d_depth = 2
        # whether to apply self-attention proposed by zhang et al. (SAGAN) inside discriminator architecture
        self.MODEL.apply_d_attn = True
        # whether to use the SN-GAN channel pattern for discriminator
        self.MODEL.d_wide = True
        # whether to use blur resample instead of average pooling for discriminator downsample
        self.MODEL.blur_resample = False
        # whether to use Concationation+Conv1x1 instead of Sum for residual skip connections
        self.MODEL.residual_concat = False
        # whether to apply moving average update for the generator
        self.MODEL.apply_g_ema = True
         # decay rate for the ema generator
        self.MODEL.g_ema_decay = 0.999
        # starting step for g_ema update
        self.MODEL.g_ema_start = 1000

        # -----------------------------------------------------------------------------
        # Loss settings
        # -----------------------------------------------------------------------------
        self.LOSS = types.SimpleNamespace()

        # loss name \in ["hinge", "non-saturating"]
        self.LOSS.type = "hinge"
        # gradient pentalty type \in ["r1", None]
        self.LOSS.grad_penalty_type = None
        # gradient penalty weight
        self.LOSS.grad_penalty_cost = 10.0

        # -----------------------------------------------------------------------------
        # Classifier pretrained settings
        # -----------------------------------------------------------------------------
        self.CLS_PRE = types.SimpleNamespace()

        # Config of the pretrained classifier used to automatically determine the model name
        self.CLS_PRE.name = "resnet20"
        self.CLS_PRE.optimizer = "sgdw"
        self.CLS_PRE.random_erasing = True
        self.CLS_PRE.extra_name = ''
        self.CLS_PRE.save_path = './save/Models/Classifiers'

        # -----------------------------------------------------------------------------
        # Run settings
        # -----------------------------------------------------------------------------
        self.RUN = types.SimpleNamespace()

        # random seed
        self.RUN.seed = 42
        # whether to save anything produced by the run (i.e. model, images, history, etc.)
        self.RUN.save = True
        # base path where to save the model
        self.RUN.save_path = './save/Models/BigGAN'
        # extra name to append to the (automatically generated) model name
        self.RUN.extra_name = ''
        # whether to use mixed precision
        self.RUN.mixed_precision = False
        # whether to compute metrics at the end of each epoch. If True, the training will be much slower
        self.RUN.compute_metrics = False
        # epoch at which start computing evaluation metrics
        self.RUN.start_eval_epoch = 0
        # whether to compute stats associated to the pretrained classifier during GAN training
        self.RUN.compute_classifier_stats = False
        # whether to reload the model (gen, disc, history, etc.) if it already exists (useful for resuming training)
        self.RUN.reload = False
        # whether to plot the images during training (useful if running in a Jupyter notebook)
        self.RUN.show_plots = False
        # whether to add a title to the plot of the generated images
        self.RUN.plot_with_title = False
        # title to add to the plot of the generated images
        self.RUN.plot_title = ''
        # whether to keep the single generated images. If False, only the GIF will be kept
        self.RUN.keep_images = False
        # whether to fix the seed at the beginning of each epoch (theoretically, should be False)
        self.RUN.fixed_seed = True


class PipelineConfiguration(Configuration):
    def __init__(self, cfg_file=None):
        super().__init__(cfg_file)

    def load_base_cfgs(self):
        print("Loading base configuration for Pipeline training")
        # -----------------------------------------------------------------------------
        # Data settings
        # -----------------------------------------------------------------------------
        self.DATA = types.SimpleNamespace()

        # dataset name \in ["FashionMNIST", "CIFAR10", "CIFAR100", "CINIC10", "DermaMNIST"]
        self.DATA.dataset = "CIFAR10"
        # image size for training
        self.DATA.image_size = 32
        # whether to convert the labels of the training set to categorical
        self.DATA.categorical = True
        # whether to convert the labels of the val/test sets to categorical
        self.DATA.val_categorical = True
        # whether to normalize the images
        self.DATA.normalize = True
        # whether to dequantize the images
        self.DATA.dequantize = False
        # whether to resize the images to img_size
        self.DATA.resize = None
        # wheter to zero pad the images to img_size
        self.DATA.padding = None
        # whether to apply random horizontal flip at dataset level
        self.DATA.horizontal_flip = False
        # whether to merge train and val sets
        self.DATA.merge_train_val = False
        # whether the last batch should be dropped in the case it has fewer samples than batch_size elements
        self.DATA.drop_remainder = False

        # -----------------------------------------------------------------------------
        # Optimizer settings
        # -----------------------------------------------------------------------------
        self.OPTIMIZATION = types.SimpleNamespace()

        # type of the optimizer \in ["adam", "sgd", "sgdw", "adabelief"]
        self.OPTIMIZATION.optimizer = "sgdw"
        # learning rate
        self.OPTIMIZATION.lr = 0.1
        # ,omentum to use for SGD/SGDW
        self.OPTIMIZATION.momentum = 0.9
        # wheter to apply Nesterov momentum for SGD/SGDW
        self.OPTIMIZATION.nesterov = True
        # weight decay to use for SGDW
        self.OPTIMIZATION.weight_decay = 1e-4
        # batch size for training
        self.OPTIMIZATION.batch_size = 128
        # batch size for evaluation
        self.OPTIMIZATION.val_batch_size = 128
        # training epochs
        self.OPTIMIZATION.epochs = 100

        # -----------------------------------------------------------------------------
        # Model settings
        # -----------------------------------------------------------------------------
        self.MODEL = types.SimpleNamespace()

        # model name \in ["resnetXX", "resnet_studiogan", "simple"]
        self.MODEL.name = "resnet20"
        # model width
        self.MODEL.width = 64

        # -----------------------------------------------------------------------------
        # Pipeline settings
        # -----------------------------------------------------------------------------
        self.PIPELINE = types.SimpleNamespace()

        # name of the GAN run to use in the pipeline. Must be specified manually
        self.PIPELINE.gan_name = None
        # list containing the desired pipeline steps \in ["all", "ckpt", "stddev", "threshold", "best"]
        self.PIPELINE.steps = ['all']
        # which checkpoints evaluate in the "checkpoint optimization" step of the pipeline.
        # A list of integers to evaluate the corresponding checkpoints, or a triplet (start, end, step) to evaluate all the checkpoints in the range [start, end] with the given step
        self.PIPELINE.ckpt_epochs = (150, 500, 10)
        # which values of standard deviation evaluate in the "stddev optimization" step of the pipeline.
        # A list of floats to evaluate the corresponding standard deviations, or a triplet (start, end, step) to evaluate all the standard deviations in the range [start, end] with the given step
        self.PIPELINE.stddev_search = (1, 2, 0.05)
        # which values of threshold evaluate in the "threshold optimization" step of the pipeline.
        # A list of floats to evaluate the corresponding thresholds, or a triplet (start, end, step) to evaluate all the thresholds in the range [start, end] with the given step
        self.PIPELINE.threshold_search = (0, 0.9, 0.1)
        # base path where GAN model is saved
        self.PIPELINE.load_path = './save/Models/BigGAN'
        # whether to apply the standing stats trick on the generator before its first use
        self.PIPELINE.apply_standing_stats = False
        # batch size to use when applying the standing stats trick (the same batch size used duing generator training works well)
        self.PIPELINE.standing_stats_bs = 192
        # how many times to try to filter the dataset before adding unfiltered samples (if filtering_attempts=-1, there is no attemps limit)
        # Useful for datasets with many classes, where the generator could be collapsed for one or more classes
        self.PIPELINE.filtering_attempts = -1
        # number of samples per class to generate for the fake dataset. If None, the number of samples per class will be computed to have the same number of samples as the real dataset
        self.PIPELINE.class_samples = None

        # -----------------------------------------------------------------------------
        # Augmentation settings
        # -----------------------------------------------------------------------------
        self.AUG = types.SimpleNamespace()

        # where to apply random horizontal flip
        self.AUG.random_flip = True
        # where to apply random crop
        self.AUG.random_crop = True
        # where to apply random rotation
        self.AUG.random_rotation = False
        # where to apply random zoom
        self.AUG.random_zoom = False
        # where to apply random erasing
        self.AUG.random_erasing = True

        # -----------------------------------------------------------------------------
        # Classifier pretrained settings
        # -----------------------------------------------------------------------------
        self.CLS_PRE = types.SimpleNamespace()

        # Config of the pretrained classifier used to automatically determine the model name
        self.CLS_PRE.name = "resnet20"
        self.CLS_PRE.optimizer = "sgdw"
        self.CLS_PRE.random_erasing = True
        self.CLS_PRE.extra_name = ''
        self.CLS_PRE.save_path = './save/Models/Classifiers'

        # -----------------------------------------------------------------------------
        # Run settings
        # -----------------------------------------------------------------------------
        self.RUN = types.SimpleNamespace()

        # random seed
        self.RUN.seed = 42
        # base path where to save the model
        self.RUN.save_path = './save/Models/BigGAN'
        # extra name to append to the (automatically generated) model name
        self.RUN.extra_name = ''
        # whether to use mixed precision
        self.RUN.mixed_precision = False


class MultiGANConfiguration(Configuration):
    def __init__(self, cfg_file=None):
        super().__init__(cfg_file)

    def load_base_cfgs(self):
        print("Loading base configuration for MultiGAN pipeline training")
        # -----------------------------------------------------------------------------
        # Data settings
        # -----------------------------------------------------------------------------
        self.DATA = types.SimpleNamespace()

        # dataset name \in ["FashionMNIST", "CIFAR10", "CIFAR100", "CINIC10", "DermaMNIST"]
        self.DATA.dataset = "CIFAR10"
        # image size for training
        self.DATA.image_size = 32
        # whether to convert the labels of the training set to categorical
        self.DATA.categorical = True
        # whether to convert the labels of the val/test sets to categorical
        self.DATA.val_categorical = True
        # whether to normalize the images
        self.DATA.normalize = True
        # whether to dequantize the images
        self.DATA.dequantize = False
        # whether to resize the images to img_size
        self.DATA.resize = None
        # wheter to zero pad the images to img_size
        self.DATA.padding = None
        # whether to apply random horizontal flip at dataset level
        self.DATA.horizontal_flip = False
        # whether to merge train and val sets
        self.DATA.merge_train_val = False
        # whether the last batch should be dropped in the case it has fewer samples than batch_size elements
        self.DATA.drop_remainder = False

        # -----------------------------------------------------------------------------
        # Optimizer settings
        # -----------------------------------------------------------------------------
        self.OPTIMIZATION = types.SimpleNamespace()

        # type of the optimizer \in ["adam", "sgd", "sgdw", "adabelief"]
        self.OPTIMIZATION.optimizer = "sgdw"
        # learning rate
        self.OPTIMIZATION.lr = 0.1
        # ,omentum to use for SGD/SGDW
        self.OPTIMIZATION.momentum = 0.9
        # wheter to apply Nesterov momentum for SGD/SGDW
        self.OPTIMIZATION.nesterov = True
        # weight decay to use for SGDW
        self.OPTIMIZATION.weight_decay = 1e-4
        # batch size for training
        self.OPTIMIZATION.batch_size = 128
        # batch size for evaluation
        self.OPTIMIZATION.val_batch_size = 128
        # training epochs
        self.OPTIMIZATION.epochs = 100

        # -----------------------------------------------------------------------------
        # Model settings
        # -----------------------------------------------------------------------------
        self.MODEL = types.SimpleNamespace()

        # model name \in ["resnetXX", "resnet_studiogan", "simple"]
        self.MODEL.name = "resnet20"
        # model width
        self.MODEL.width = 64

        # -----------------------------------------------------------------------------
        # Pipeline settings
        # -----------------------------------------------------------------------------
        self.PIPELINE = types.SimpleNamespace()

        # names of the GANs run to use in the pipeline. Must be specified manually
        self.PIPELINE.gan_names = None
        # base path where GAN model is saved
        self.PIPELINE.load_path = './save/Models/BigGAN'
        # whether to use only one GAN at each trainig epoch (in a circular way), or all GANs at each epoch
        self.PIPELINE.one_gan_for_epoch = False
        # whether to apply the standing stats trick on the generator before its first use
        self.PIPELINE.apply_standing_stats = False
        # batch size to use when applying the standing stats trick (the same batch size used duing generator training works well)
        self.PIPELINE.standing_stats_bs = 192
        # how many times to try to filter the dataset before adding unfiltered samples (if filtering_attempts=-1, there is no attemps limit)
        # Useful for datasets with many classes, where the generator could be collapsed for one or more classes
        self.PIPELINE.filtering_attempts = -1
        # number of samples per class to generate for the fake dataset. If None, the number of samples per class will be computed to have the same number of samples as the real dataset
        self.PIPELINE.class_samples = None
        # number of samples per class to generate for the fake dataset during the pipeline steps. If None, the number of samples per class will be computed to have the same number of samples as the real dataset
        # Useful to load the best hyperparameters from a pipeline trained with more/less samples than the one that will be used for the current MultiGAN pipeline
        self.PIPELINE.best_class_samples = None
        # extra name to append to the (automatically generated) classifier name used in the 'best hyperparameters' step
        self.PIPELINE.best_extra_name = ''

        # -----------------------------------------------------------------------------
        # Augmentation settings
        # -----------------------------------------------------------------------------
        self.AUG = types.SimpleNamespace()

        # where to apply random horizontal flip
        self.AUG.random_flip = True
        # where to apply random crop
        self.AUG.random_crop = True
        # where to apply random rotation
        self.AUG.random_rotation = False
        # where to apply random zoom
        self.AUG.random_zoom = False
        # where to apply random erasing
        self.AUG.random_erasing = True

        # -----------------------------------------------------------------------------
        # Classifier pretrained settings
        # -----------------------------------------------------------------------------
        self.CLS_PRE = types.SimpleNamespace()

        # Config of the pretrained classifier used to automatically determine the model name
        self.CLS_PRE.name = "resnet20"
        self.CLS_PRE.optimizer = "sgdw"
        self.CLS_PRE.random_erasing = True
        self.CLS_PRE.extra_name = ''
        self.CLS_PRE.save_path = './save/Models/Classifiers'

        # -----------------------------------------------------------------------------
        # Run settings
        # -----------------------------------------------------------------------------
        self.RUN = types.SimpleNamespace()

        # random seed
        self.RUN.seed = 42
        # base path where to save the model
        self.RUN.save_path = './save/Models/BigGAN'
        # extra name to append to the (automatically generated) model name
        self.RUN.extra_name = ''
        # whether to use mixed precision
        self.RUN.mixed_precision = False