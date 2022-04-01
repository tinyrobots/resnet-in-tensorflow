# Train Resnet on Bumpworld gloss classification. Modified from:
# https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================

'''
NOTE: if using Test code as standalone,
need to change ckpt flag and paths in hyper_parameters.py
to match network you want to test.
'''
# from cifar10_input import * # KS: get error from cv2 if tf has already been imported
import os
from resnet import *
from datetime import datetime
import time
import pandas as pd
from skimage import transform, io
from sklearn.metrics import accuracy_score

# bumpworld data pointers
img_dir = '../../../data/continuous_1Dgloss_10k/'
max_scene = 9500
test_scene = 9000
scene_data = range(test_scene,max_scene) # how many/which of your images you want to use here (CIFAR = 50k train)
train_cutoff = 0 # index at which to switch from training to validation images

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_DEPTH = 3
NUM_CLASS = 2

from tensorflow import logging
logging.set_verbosity(logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ################################################################################
# Helper functions

def prepare_all_data():
    '''
    Read all the train data (both images and labels) into numpy arrays.

    Modified version of prepare_train_data() in cifar10_input.py
    '''

    # Load my BUMPWORLD images at the specified resolution
    desired_im_sz = (IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH)
    splits = {'scene': scene_data}
    im_dir = os.path.join(img_dir, 'rgb/')
    _, _, all_frames = os.walk(im_dir).next() # warning: .next() doesn't work in Python3
    all_frames = sorted(all_frames)
    for split in splits:
        im_list = all_frames[splits[split][0] : splits[split][len(splits[split])-1]+1]
        source_list = [str(i) for i in splits[split]]
        data = np.zeros((len(im_list),) + desired_im_sz, np.float32)
        for i, im_file in enumerate(im_list):
            im = io.imread(os.path.join(im_dir, im_file)) # loads as uint8 (0-255 integers)?
            im = transform.resize(im[:,:,0:-1], desired_im_sz) # explicitly cut off alpha channel here, or get weird colour behav.
            im = np.float32(255*im) # the resize function changes dtype to float64 in range [0, 1] - not OK for make_feed_dict!
            data[i] = im
    print('Bumpworld images loaded')
    print(data.shape)

    # load label data
    # Fetch the .csv file that provides information about each of the scenes
    scene_log = pd.read_csv(os.path.join(img_dir,'scene_log.csv'))
    print(scene_log.head()) # sanity checking

    scene_min = test_scene
    scene_max = max_scene
    factors = ['gloss_cat']

    probe_scenes = ['scene'+str(s).zfill(6) for s in range(scene_min+1, scene_max+1)]
    # create an ordered dataframe for all the scenes we have in the net data, with scene info for the available ones
    probe_log = pd.DataFrame(columns=['scene_num','gloss_cat'])
    probe_log['scene_num'] = probe_scenes # add probe IDs for all, even ones we don't have scene info for

    # Then fill in all of those which have scene info from the log, keeping NaNs otherwise
    # (probably way faster more elegant ways to do this dataframe magic!)
    for row in range(len(probe_log)):
        if not scene_log[scene_log['scene_num'] == probe_log.loc[row,'scene_num']].empty:
            # print(row)
            oldrow = scene_log[scene_log['scene_num']==probe_log.loc[row,'scene_num']].index.item() # get row with corresponding scene ID
            probe_log.loc[row, ['scene_num',factors[0]]] =  scene_log.loc[oldrow,['scene_num',factors[0]]]

    # If there are any unidentifiable scenes (ie those with no info), remove these now from both the probe log and the data arrays
    nan_idx = list(probe_log[probe_log[factors[0]].isnull()].index) # just picking light_angle as the first var
    # X_hidden = np.delete(X_hidden, nan_idx, axis=0) # X_hidden already doesn't contain the NaN row!
    probe_log = probe_log.dropna() # also drop them from the info log

    labels = np.array(probe_log['gloss_cat'])
    labels = labels.astype(np.int32) # reformat for tensorflow

    return data, labels
# ################################################################################
# Define model class:

class Train(object):
    '''
    This Object is responsible for all the training and validation process
    '''
    def __init__(self):
        # Set up all the placeholders
        self.placeholders()


    def placeholders(self):
        '''
        There are five placeholders in total.
        image_placeholder and label_placeholder are for train images and labels
        vali_image_placeholder and vali_label_placeholder are for validation imgaes and labels
        lr_placeholder is for learning rate. Feed in learning rate each time of training
        implements learning rate decay easily
        '''
        self.image_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.train_batch_size, IMG_HEIGHT,
                                                        IMG_WIDTH, IMG_DEPTH])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size])

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,
                                                                IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.validation_batch_size])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])



    def build_train_validation_graph(self):
        '''
        This function builds the train graph and validation graph at the same time.

        '''
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        logits = inference(self.image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        vali_logits = inference(self.vali_image_placeholder, FLAGS.num_residual_blocks, reuse=True)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(logits, self.label_placeholder)
        self.full_loss = tf.add_n([loss] + regu_losses)

        predictions = tf.nn.softmax(logits)
        self.train_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)


        # Validation loss
        self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        vali_predictions = tf.nn.softmax(vali_logits)
        self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)

        self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss,
                                                                self.train_top1_error)
        self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)



    def train(self):
        '''
        This is the main function for training
        '''

        print('Entering main training loop...')
        # For the first step, we are loading all training images and validation images into the
        # memory
        train_and_test_data, train_and_test_labels = prepare_all_data()
        train_data = train_and_test_data[:train_cutoff, :, :]
        test_data = train_and_test_data[train_cutoff:, :, :]
        train_labels = train_and_test_labels[:train_cutoff]
        test_labels = train_and_test_labels[train_cutoff:]

        # quick messy printouts for sanity checking
        print('Image data check:')
        print(train_data[0,:,:,2])
        print('Labels check:')
        print(train_labels[0:100])

        # Build the graph for train and validation
        self.build_train_validation_graph()

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()
        sess = tf.Session()


        # If you want to load from a checkpoint
        if FLAGS.is_use_ckpt is True:
            saver.restore(sess, FLAGS.ckpt_path)
            print 'Restored from checkpoint...'
        else:
            sess.run(init)

        # This summary writer object helps write summaries on tensorboard
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)


        # These lists are used to save a csv file at last
        step_list = []
        train_error_list = []
        val_error_list = []

        print 'Start training...'
        print '----------------------------'

        for step in xrange(FLAGS.train_steps):

            train_batch_data, train_batch_labels = self.generate_augment_train_batch(train_data, train_labels,
                                                                        FLAGS.train_batch_size)


            validation_batch_data, validation_batch_labels = self.generate_vali_batch(test_data,
                                                           test_labels, FLAGS.validation_batch_size)

            # Want to validate once before training. You may check the theoretical validation
            # loss first
            if step % FLAGS.report_freq == 0:

                if FLAGS.is_full_validation is True:
                    validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss,
                                            top1_error=self.vali_top1_error, vali_data=vali_data,
                                            vali_labels=vali_labels, session=sess,
                                            batch_data=train_batch_data, batch_label=train_batch_labels)

                    vali_summ = tf.Summary()
                    vali_summ.value.add(tag='full_validation_error',
                                        simple_value=validation_error_value.astype(np.float))
                    summary_writer.add_summary(vali_summ, step)
                    summary_writer.flush()

                else:
                    _, validation_error_value, validation_loss_value = sess.run([self.val_op,
                                                                     self.vali_top1_error,
                                                                 self.vali_loss],
                                                {self.image_placeholder: train_batch_data,
                                                 self.label_placeholder: train_batch_labels,
                                                 self.vali_image_placeholder: validation_batch_data,
                                                 self.vali_label_placeholder: validation_batch_labels,
                                                 self.lr_placeholder: FLAGS.init_lr})

                val_error_list.append(validation_error_value)


            start_time = time.time()

            _, _, train_loss_value, train_error_value = sess.run([self.train_op, self.train_ema_op,
                                                           self.full_loss, self.train_top1_error],
                                {self.image_placeholder: train_batch_data,
                                  self.label_placeholder: train_batch_labels,
                                  self.vali_image_placeholder: validation_batch_data,
                                  self.vali_label_placeholder: validation_batch_labels,
                                  self.lr_placeholder: FLAGS.init_lr})
            duration = time.time() - start_time


            if step % FLAGS.report_freq == 0:
                summary_str = sess.run(summary_op, {self.image_placeholder: train_batch_data,
                                                    self.label_placeholder: train_batch_labels,
                                                    self.vali_image_placeholder: validation_batch_data,
                                                    self.vali_label_placeholder: validation_batch_labels,
                                                    self.lr_placeholder: FLAGS.init_lr})
                summary_writer.add_summary(summary_str, step)

                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print format_str % (datetime.now(), step, train_loss_value, examples_per_sec,
                                    sec_per_batch)
                print 'Train top1 error = ', train_error_value
                print 'Validation top1 error = %.4f' % validation_error_value
                print 'Validation loss = ', validation_loss_value
                print '----------------------------'

                step_list.append(step)
                train_error_list.append(train_error_value)



            if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
                FLAGS.init_lr = 0.1 * FLAGS.init_lr
                print 'Learning rate decayed to ', FLAGS.init_lr

            # Save checkpoints every 10000 steps
            if step % 1000 == 0 or (step + 1) == FLAGS.train_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                df = pd.DataFrame(data={'step':step_list, 'train_error':train_error_list,
                                'validation_error': val_error_list})
                df.to_csv(train_dir + FLAGS.version + '_error.csv')


    def test(self, test_image_array):
        '''
        This function is used to evaluate the test data. Please finish pre-precessing in advance

        :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
        img_depth]
        :return: the softmax probability with shape [num_test_images, num_labels]
        '''
        num_test_images = len(test_image_array)
        num_batches = num_test_images // FLAGS.test_batch_size
        remain_images = num_test_images % FLAGS.test_batch_size
        print '%i test batches in total...' %num_batches

        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                        IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        # Build the test graph
        logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        predictions = tf.nn.softmax(logits)

        # Initialize a new session and restore a checkpoint
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()

        saver.restore(sess, FLAGS.test_ckpt_path)
        print 'Model restored from ', FLAGS.test_ckpt_path

        prediction_array = np.array([]).reshape(-1, NUM_CLASS)
        # Test by batches
        for step in range(num_batches):
            if step % 10 == 0:
                print '%i batches finished!' %step
            offset = step * FLAGS.test_batch_size
            test_image_batch = test_image_array[offset:offset+FLAGS.test_batch_size, ...]

            batch_prediction_array = sess.run(predictions,
                                        feed_dict={self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        # If test_batch_size is not a divisor of num_test_images
        if remain_images != 0:
            self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                        IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
            # Build the test graph
            logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=True)
            predictions = tf.nn.softmax(logits)

            test_image_batch = test_image_array[-remain_images:, ...]

            batch_prediction_array = sess.run(predictions, feed_dict={
                self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        return prediction_array



    ## Helper functions
    def loss(self, logits, labels):
        '''
        Calculate the cross entropy loss given logits and true labels
        :param logits: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size]
        :return: loss tensor with shape [1]
        '''
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean


    def top_k_error(self, predictions, labels, k):
        '''
        Calculate the top-k error
        :param predictions: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k: int
        :return: tensor with shape [1]
        '''
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)


    def generate_vali_batch(self, vali_data, vali_label, vali_batch_size):
        '''
        If you want to use a random batch of validation data to validate instead of using the
        whole validation data, this function helps you generate that batch
        :param vali_data: 4D numpy array
        :param vali_label: 1D numpy array
        :param vali_batch_size: int
        :return: 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice((vali_data.shape[0] - vali_batch_size), 1)[0]
        vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
        vali_label_batch = vali_label[offset:offset+vali_batch_size]
        return vali_data_batch, vali_label_batch


    def generate_augment_train_batch(self, train_data, train_labels, train_batch_size):
        '''
        This function helps generate a batch of train data, and random crop, horizontally flip
        and whiten them at the same time
        :param train_data: 4D numpy array
        :param train_labels: 1D numpy array
        :param train_batch_size: int
        :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice((train_data.shape[0] - train_batch_size), 1)[0]
        batch_data = train_data[offset:offset+train_batch_size, ...]
        # batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size) # KS: disable augmentation, as not used in PixelVAE
        # batch_data = whitening_image(batch_data) # KS: for now, don't use whitening, since not used in PixelVAE
        batch_label = train_labels[offset:offset+FLAGS.train_batch_size]

        return batch_data, batch_label


    def train_operation(self, global_step, total_loss, top1_error):
        '''
        Defines train operations
        :param global_step: tensor variable with shape [1]
        :param total_loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :return: two operations. Running train_op will do optimization once. Running train_ema_op
        will generate the moving average of train error and train loss for tensorboard
        '''
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', total_loss)
        tf.summary.scalar('train_top1_error', top1_error)

        # The ema object help calculate the moving average of train loss and train error
        ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])
        tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
        tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op


    def validation_op(self, validation_step, top1_error, loss):
        '''
        Defines validation operations
        :param validation_step: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param loss: tensor with shape [1]
        :return: validation operation
        '''

        # This ema object help calculate the moving average of validation loss and error

        # ema with decay = 0.0 won't average things at all. This returns the original error
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)


        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        top1_error_val = ema.average(top1_error)
        top1_error_avg = ema2.average(top1_error)
        loss_val = ema.average(loss)
        loss_val_avg = ema2.average(loss)

        # Summarize these values on tensorboard
        tf.summary.scalar('val_top1_error', top1_error_val)
        tf.summary.scalar('val_top1_error_avg', top1_error_avg)
        tf.summary.scalar('val_loss', loss_val)
        tf.summary.scalar('val_loss_avg', loss_val_avg)
        return val_op


    def full_validation(self, loss, top1_error, session, vali_data, vali_labels, batch_data,
                        batch_label):
        '''
        Runs validation on all the valdiation images
        :param loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param session: the current tensorflow session
        :param vali_data: 4D numpy array
        :param vali_labels: 1D numpy array
        :param batch_data: 4D numpy array. training batch to feed dict and fetch the weights
        :param batch_label: 1D numpy array. training labels to feed the dict
        :return: float, float
        '''
        num_batches = vali_data.shape[0] // FLAGS.validation_batch_size
        order = np.random.choice(vali_data.shape[0], num_batches * FLAGS.validation_batch_size)
        vali_data_subset = vali_data[order, ...]
        vali_labels_subset = vali_labels[order]

        loss_list = []
        error_list = []

        for step in range(num_batches):
            offset = step * FLAGS.validation_batch_size
            feed_dict = {self.image_placeholder: batch_data, self.label_placeholder: batch_label,
                self.vali_image_placeholder: vali_data_subset[offset:offset+FLAGS.validation_batch_size, ...],
                self.vali_label_placeholder: vali_labels_subset[offset:offset+FLAGS.validation_batch_size],
                self.lr_placeholder: FLAGS.init_lr}
            loss_value, top1_error_value = session.run([loss, top1_error], feed_dict=feed_dict)
            loss_list.append(loss_value)
            error_list.append(top1_error_value)

        return np.mean(loss_list), np.mean(error_list)


# Initialize the Train object
train = Train()
# instead of training, only test
train_and_test_data, train_and_test_labels = prepare_all_data()
test_data = train_and_test_data[train_cutoff:, :, :]
test_labels = train_and_test_labels[train_cutoff:]
print(test_data.shape)
predictions = train.test(test_data) # predictions is the predicted softmax array.
pred_error = train.top_k_error(tf.convert_to_tensor(predictions, dtype=tf.float32), test_labels, 1)
for i in range(len(test_labels)):
    print('True label: {} | Predicted label: {}'.format(test_labels[i], np.round(predictions[i])))
print('Accuracy on full validation set: {}'.format(1-tf.Session().run(pred_error)))
