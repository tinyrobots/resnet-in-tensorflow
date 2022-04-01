# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

## The following flags are related to save paths, tensorboard outputs and screen outputs

# KS make sure model name is correct, and is_use_ckpt=True only if you want to probe
model_name = 'resnet_regression_bn_debug'
train_dir = './logs_{}/'.format(model_name)# + FLAGS.version + '/'
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

ckpt = 1000
tf.app.flags.DEFINE_string('version', model_name, '''Model name defining the directory to save
logs and checkpoints''')
## If you want to load a checkpoint and probe / continue training
tf.app.flags.DEFINE_boolean('is_use_ckpt', True, '''Whether to load a checkpoint and continue
training''')
tf.app.flags.DEFINE_string('ckpt_path', '{}model.ckpt-{}'.format(train_dir, ckpt), '''Checkpoint
directory to restore''')
tf.app.flags.DEFINE_string('test_ckpt_path', '{}model.ckpt-{}'.format(train_dir, ckpt), '''Checkpoint
directory to restore''')


tf.app.flags.DEFINE_integer('report_freq', 20, '''Steps takes to output errors on the screen
and write summaries''')
tf.app.flags.DEFINE_float('train_ema_decay', 0.95, '''The decay factor of the train error's
moving average shown on tensorboard''')


## The following flags define hyper-parameters regards training

tf.app.flags.DEFINE_integer('train_steps', 1001, '''Total steps that you want to train''')
tf.app.flags.DEFINE_boolean('is_full_validation', False, '''Validation w/ full validation set or
a random batch''')
tf.app.flags.DEFINE_integer('train_batch_size', 32, '''Train batch size''')
tf.app.flags.DEFINE_integer('validation_batch_size', 32, '''Validation batch size, better to be
a divisor of 10000 for this task''')
tf.app.flags.DEFINE_integer('test_batch_size', 32, '''Test batch size''') # make sure is divisor of probe set

tf.app.flags.DEFINE_float('init_lr', 0.0001, '''Initial learning rate''')
tf.app.flags.DEFINE_float('lr_decay_factor', 0.99999, '''How much to decay the learning rate each
time''')
tf.app.flags.DEFINE_integer('smooth_decay', True, '''Whether to apply decay at every step''')
tf.app.flags.DEFINE_integer('decay_step0', 1000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step1', 4000, '''At which step to decay the learning rate''')


## The following flags define hyper-parameters modifying the training network

tf.app.flags.DEFINE_integer('num_residual_blocks', 3, '''How many residual blocks do you want''')
tf.app.flags.DEFINE_float('weight_decay', 0.000001, '''scale for l2 regularization''') # 0.0002 for all categorical models


## The following flags are related to data-augmentation # KS: don't want data aug. for Bumpworld

# tf.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero padding on
# each side of the image''')
