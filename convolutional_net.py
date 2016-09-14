import tensorflow as tf
import joblib
import cPickle as pk
import os
import numpy as np

sess = tf.InteractiveSession()


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def load_kaggle():

    data_dir = '/Users/Tofigh/Documents/Kaggle/train/features'

    fd = open(os.path.join(data_dir, 'subj10_series1_features_ch0_white'))
    ch = 0
    trX = joblib.load(fd).astype(np.float)
    [N, nx, nx] = trX.shape
    trX = trX.reshape((N, nx, nx, 1))
    maxNorm = np.max(np.abs(trX))
    trX = trX / maxNorm
    fd = open(os.path.join(data_dir, 'subj10_series1_labels'))
    trY = np.array(pk.load(fd)).astype(np.float)
    ind_zeros = np.where(trY == 0)[0]
    seed_select = 0
    np.random.seed(seed_select)
    selected_zeros = np.random.choice(ind_zeros, 1000, replace=False)
    trY_zeros = trY[selected_zeros]
    ind_nonzeros = np.where(trY > 0)[0]
    trY = np.concatenate((trY_zeros, trY[ind_nonzeros]))
    trX = np.concatenate((trX[selected_zeros, :, :, :], trX[ind_nonzeros, :, :, :]), axis=0)
    # fd = open(os.path.join(data_dir, 'subj10_series7_features'))
    # teX = joblib.load(fd).astype(np.float)[:, :, :, ch]
    # [N, nx, nx] = teX.shape
    # teX = teX.reshape((N, nx, nx, 1))
    # maxNorm = np.max(teX)
    # meanNorm = np.mean(teX, axis=0)
    # teX = (teX - meanNorm) / maxNorm
    #
    # fd = open(os.path.join(data_dir, 'subj10_series7_labels'))
    # teY = np.array(pk.load(fd)).astype(np.float)

    trY = np.asarray(trY)
    # teY = np.asarray(teY)

    # X = np.concatenate((trX, teX), axis=0)
    # y = np.concatenate((trY, teY), axis=0)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(trX)
    np.random.seed(seed)
    np.random.shuffle(trY)
    y_vec = np.zeros((len(trY), 7), dtype=np.float)
    for i, label in enumerate(trY):
        y_vec[i, label] = 1.0
    print "YAAAA I read it.***************"
    print trX.shape
    return trX, y_vec

num_labels = 7

x_image = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[None, num_labels])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, num_labels])
b_fc2 = bias_variable([num_labels])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

batch_size = 64
l_rate = 0.00005

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(l_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())


X_data, y_data = load_kaggle()
N = X_data.shape[0]
num_batch = N/batch_size
for epoch in range(200):
    for batch in iterate_minibatches(X_data, y_data, batch_size, shuffle= False):
        train_step.run(feed_dict={x_image: batch[0], y_: batch[1], keep_prob: 0.5})

    train_accuracy = accuracy.eval(feed_dict={
        x_image: batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g" % (epoch, train_accuracy))

