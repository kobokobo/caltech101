import tensorflow as tf
import numpy as np
# from tensorflow.python import debug as tf_debug

# specify category
categories = ["chair","cup","Faces_easy","lamp","laptop","scissors","soccer_ball","stapler","watch","yin_yang"]


nb_classes = len(categories)
# image size
image_w = 64
image_h = 64

# data loading
X_train, X_test, y_train, y_test = np.load("/data/101_Caltech_npy/10obj.npy")
# data normalization
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float")  / 256
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)

nums = 10 # 10 class(0-9 digit)

# declare of  placeholder
x  = tf.placeholder(tf.float32, shape=(None, image_h, image_w, 3), name="x") # image data
y_ = tf.placeholder(tf.float32, shape=(None, nums), name="y_")  # label (teaching data)

# Return a total of `num` random samples and labels.
def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# initialize for weight and bais [-0.1 ~ 0.1] gauss distribution
def weight_variable(name, shape):
    W_init = tf.truncated_normal(shape, stddev=0.1)
    W = tf.Variable(W_init, name="W_"+name)
    return W

def bias_variable(name, size):
    b_init = tf.constant(0.1, shape=[size])
    b = tf.Variable(b_init, name="b_"+name)
    return b

# convolution function
def conv2d(x, W):
    # strides=[1,1,1,1]
    # HWCN: 1 stride on image hight, 1 stride on image width, 1 stride on image channel, 1 stride on image data set
    # "padding SAME" is zero padding
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# max pooling function
def max_pool(x):
    # in case of ksize = [1, 3, 3, 1] and strides = [1, 2, 2, 1]
    # pick up pixel with max value from 3x3 region and stride 2 make the output of image size half.
    # finding the value from large area, so it make more robust against offset noise.
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# convolution layer 1
with tf.name_scope('conv1') as scope:
    W_conv1 = weight_variable('conv1', [3, 3, 3, 32])
    b_conv1 = bias_variable('conv1', 32)
    # input x is transferred to x_image which is 4 dimension tensor [batch, in_height, in_width, in_channels]
    x_image = tf.reshape(x, [-1, 64, 64, 3])
    # W_conv1::4 dimension tensor [filter_height, filter_width, in_channels, out_channels ]
    # relu has sparse effect and make more faster learning
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# pooling layer 1
with tf.name_scope('pool1') as scope:
    h_pool1 = max_pool(h_conv1)

# convolution layer 2
with tf.name_scope('conv2') as scope:
    W_conv2 = weight_variable('conv2', [3, 3, 32, 64])
    b_conv2 = bias_variable('conv2', 64)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# convolution layer 3
with tf.name_scope('conv3') as scope:
    W_conv3 = weight_variable('conv3', [3, 3, 64, 64])
    b_conv3 = bias_variable('conv3', 64)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

# convolution layer 4
with tf.name_scope('conv4') as scope:
    W_conv4 = weight_variable('conv4', [3, 3, 64, 64])
    b_conv4 = bias_variable('conv4', 64)
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

# pooling layer 4
with tf.name_scope('pool4') as scope:
    h_pool4 = max_pool(h_conv4)

# fully connected layer
with tf.name_scope('fully_connected') as scope:
    # 2 times Pooling, and the image size is 64->32->16
    # 64 number of filters
    n = 16 * 16 * 64
    # n = 16 * 16 * 64 (in), 1024 number of neuron (out)
    W_fc = weight_variable('fc', [n, 1024])
    b_fc = bias_variable('fc', 1024)
    h_pool4_flat = tf.reshape(h_pool4, [-1, n])
    # matmul is matrix A * matrix B
    h_fc = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc) + b_fc)
    # h_fc_drop = tf.nn.dropout(h_fc, 0.25)

# reading layer
with tf.name_scope('readout') as scope:
    W_fc2 = weight_variable('fc2', [1024, nums])
    b_fc2 = bias_variable('fc2', nums)
    y_conv = tf.nn.softmax(tf.matmul(h_fc, W_fc2) + b_fc2)

# loss function and training of the network
with tf.name_scope('loss') as scope:
    mean_square_error = tf.reduce_sum(tf.pow(y_conv - y_, 2) / (2.0 * tf.cast(tf.shape(y_)[0], tf.float32)))
    tf.summary.scalar('loss', mean_square_error)

with tf.name_scope('training') as scope:
    # optimizer = tf.train.AdamOptimizer(0.00007)
    optimizer = tf.train.RMSPropOptimizer(0.00007)
    train_step = optimizer.minimize(mean_square_error)

# evaluation of the network
with tf.name_scope('predict') as scope:
    # setting 1 to the second param, find the max row number in each column
    predict_step = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    # reduce_mean: calculate the mean value in all the array
    accuracy_step = tf.reduce_mean(tf.cast(predict_step, tf.float32))
    tf.summary.scalar('accuracy', accuracy_step)

# For display
summary = tf.summary.merge_all()

# setting of feed_dict (parameter)
def set_feed(images, labels):
    return {x: images, y_: labels}

# start session
with tf.Session() as sess:
    # tf.Variable( ) need initialization
    sess.run(tf.global_variables_initializer())
    # debug
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    # for TensorBoard
    tw = tf.summary.FileWriter('log_dir', graph=sess.graph)
    # create the feed for test
    test_fd = set_feed(X_test, y_test)

    # start training
    for step in range(10000):
        # debug_param = sess.run([h_conv1], feed_dict=fd)
        # print("xxxxxxxxxxx_debug_xxxxxxxxxxx=", debug_param)
        Xtr, Ytr = next_batch(32, X_train, y_train)
        fd = set_feed(Xtr, Ytr)
        _, loss = sess.run([train_step, mean_square_error], feed_dict=fd)
        if step % 50 == 0:
            acc, w_summary = sess.run([accuracy_step, summary], feed_dict=test_fd)
            print("step=", step, "loss=", loss, "acc=", acc)
            tw.add_summary(w_summary, step)

    # show the final result
    acc = sess.run(accuracy_step, feed_dict=test_fd)
    print("accuracy rate=", acc)




