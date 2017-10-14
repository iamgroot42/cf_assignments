
# coding: utf-8

# In[26]:
from __future__ import division, print_function, absolute_import


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.platform import flags
import parser 

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_folder', './ml-100k/', 'Path to Movielens 100k/1m dataset')
flags.DEFINE_integer('bsize', 16, 'Batch size')
flags.DEFINE_integer('num_iters', 10000, 'Number of iterations')
flags.DEFINE_integer('show_every', 10, 'Show trainign and testing accuracy after every X iterations')
flags.DEFINE_float('lr', 0.01, 'Learning rate')
flags.DEFINE_float('reg_lambda', 0.1, "Lambda for regularization")
flags.DEFINE_string('aetype', 'u', 'Whether autoencoder is user-based(u) or item based(u)')

#Training Parameters
learning_rate = FLAGS.lr
num_iters = FLAGS.num_iters
batch_size = FLAGS.bsize
lamb = FLAGS.reg_lambda
dd = FLAGS.dataset_folder
show_every = FLAGS.show_every


# Don't hog GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


# In[27]:


def mae(y, y_):
    flaty = y.flatten()
    flaty_ = y_.flatten()
    mask = (flaty != 0)
    relevant_count = np.sum(mask)
    return np.sum(np.multiply(mask, np.absolute(flaty-flaty_)))/relevant_count


# In[28]:


def autorec(num_input, num_hidden, data_generator, test_data):
    X = tf.placeholder("float", [None, R.shape[1]])

    print("K is %d" % (num_hidden))
    
    weights = {
        'encoder': tf.Variable(tf.random_normal([num_input, num_hidden])),
        'decoder': tf.Variable(tf.random_normal([num_hidden, num_input])),
    }
    biases = {
        'encoder': tf.Variable(tf.random_normal([num_hidden])),
        'decoder': tf.Variable(tf.random_normal([num_input])),
    }

    def encoder(x):
        layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder']), biases['encoder']))
        mask = tf.cast(tf.not_equal(tf.constant(0, dtype=tf.float32), x), tf.float32)
        return layer, mask

    def decoder(x, mask):
        layer = 2.5*(1+tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder']), biases['decoder'])))
        masked_backfunc = tf.multiply(layer, mask)
        # Trick to allow back-gradient propagataion for only part of input nodes
        back_masked_layer = masked_backfunc + tf.stop_gradient(layer - masked_backfunc) 
        return layer
    
    encoder_op, mask = encoder(X)
    decoder_op = decoder(encoder_op, mask)

    y_pred = decoder_op
    y_true = X
    
    # Normal MSE loss for autoencoder:
    loss = tf.reduce_mean(tf.where(y_true==0, tf.zeros_like(y_true), tf.pow(y_true - y_pred, 2)))
    # Regularization term:
    loss += (lamb/2) * (tf.nn.l2_loss(weights['encoder']) + tf.nn.l2_loss(weights['decoder']))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    init = tf.global_variables_initializer()    
    sess.run(init)
    
    # Train model
    for i in range(num_iters):
        batch_x = data_generator.next()
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        if i%show_every == 0:
            print("Training loss (MSE) : %.4f" % (l))
    
    # Test model and report MAE
    test_data_pred = sess.run(decoder_op, feed_dict={X: test_data})
    return mae(test_data, test_data_pred)


# In[29]:


movies = parser.load_data(dd + "u.item", parser.Movie, '|')
ratings = []

for i in range(1,1+5):
    ratings.append( [parser.load_data(dd + "u" + str(i) + ".base", parser.Rating, '\t'), parser.load_data(dd + "u" + str(i) + ".test", parser.Rating, '\t')] )
users = parser.load_data(dd + "u.user", parser.User, '|')

hidden = [10, 20, 40, 80, 100, 200, 300, 400, 500]
hidden_errors = []

for num_hidden in hidden:
    errors = []
    for fold in ratings:
        R = parser.R_matrix(len(users), len(movies), fold[0]).astype('float32')
        R_test = parser.R_matrix(len(users), len(movies), fold[1]).astype('float32')
    
	if FLAGS.aetype == 'u':
		R = R.T
		R_test = R_test.T
        data_generator = parser.get_next_batch(R, batch_size)
    
        error = autorec(R.shape[1], num_hidden, data_generator, R_test)
        print("MAE for this validation: %f" % (error))
        errors.append(error)
    print("MAE for all folds: %f" % (sum(errors)/float(len(errors))))
    hidden_errors.append(sum(errors)/float(len(errors)))

# Plot MAE with varying number of hidden nodes
plt.plot(hidden, hidden_errors)
plt.title('Variation of MAE with number of hidden units')
plt.ylabel('MAE')
plt.xlabel('NUmber of hidden units')
plt.legend()
plt.savefig('graph.png')
