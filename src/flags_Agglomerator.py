from absl import flags

flags.DEFINE_string('dataset', 'CIFAR10', 'Dataset name.')
flags.DEFINE_string('exp_name', 'CIFAR10', 'Experiment name.')

# NETWORK PARAMETERS

flags.DEFINE_integer('patch_size', 1, 'Patch size.')
flags.DEFINE_integer('patch_dim', 128, 'Patch feature dimension.')
flags.DEFINE_integer('batch_size', 1024, 'Batch size.')
flags.DEFINE_integer('levels', 2, 'Columns levels.')
flags.DEFINE_bool('supervise', False, 'Supervise training.')
flags.DEFINE_integer('image_size', 32, 'Input images size.')
flags.DEFINE_integer('conv_image_size', 8, 'Conv images size.')
flags.DEFINE_integer('n_channels', 3, 'Number of image channels.')
flags.DEFINE_integer('n_classes', 10, 'Number of classes.')
flags.DEFINE_integer('iters', None, 'Number of iterations for the columns (if None it will be set by the network).')
flags.DEFINE_integer('denoise_iter', -1, 'At which iteration to perform denoising.')
flags.DEFINE_float('dropout', 0.3, 'Dropout.')
flags.DEFINE_float('temperature', 0.07, 'Contrastive temperature.')
flags.DEFINE_integer('contr_dim', 512, 'Contrastive hidden dimension.')

# TRAINING PARAMETERS

flags.DEFINE_string('mode', 'train', 'train/test/freeze.')
flags.DEFINE_float('learning_rate', 0.05, 'Learning rate.')
flags.DEFINE_float('lr_speed', 2000, 'Learning rate scheduler speed.')
flags.DEFINE_boolean('resume_training', False,
                     'Resume training using a checkpoint.')
flags.DEFINE_string('load_checkpoint_dir', 'path_to_checkpoint.ckpt',
                    'Load previous existing checkpoint.')
flags.DEFINE_integer('seed', 42, 'Seed.')
flags.DEFINE_integer('max_epochs', 10000, 'Number of training epochs.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight decay.')
flags.DEFINE_integer('num_workers', 8, 'Number of workers.')
flags.DEFINE_integer('num_gpus', 1, 'Number of gpus.')

flags.DEFINE_float('limit_train', 1.0, 'Limit train set.')
flags.DEFINE_float('limit_val', 1.0, 'Limit val set.')
flags.DEFINE_float('limit_test', 1.0, 'Limit test set.')


flags.DEFINE_bool('plot_islands', False, 'Plot islands of agreement.')