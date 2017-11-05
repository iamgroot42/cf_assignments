import parser
import numpy as np
import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


def find_reprs(X_test, X_train, lambda_val = 1, learning_rate=1):
	X = tf.placeholder("float", X_test.shape)

	D = tf.placeholder("float", (X_train.shape[1], X_train.shape[0]))
	R = tf.placeholder("float", (X_train.shape[0], X_test.shape[0]))

	R_weights = tf.Variable(tf.random_normal([X_train.shape[0], X_test.shape[0]]))
	
	minimization_term = tf.norm(X - tf.transpose(tf.matmul(D, R_weights)), ord='fro', axis=(0,1))
	regularization_term = lambda_val * tf.cast(tf.count_nonzero(R_weights), tf.float32)

	loss = minimization_term + regularization_term
	loss /= X_test.shape[0]
	optimizer = tf.train.AdadeltaOptimizer(learning_rate)
	
	R_new = optimizer.minimize(loss, var_list=[R_weights])
	
	init = tf.global_variables_initializer()
	sess.run(init)
	
	prev_error = np.inf
	max_epochs = 1000

	# Train model
	R_train = None
	for i in range(max_epochs):

		# Update R
		_, l, R_train = sess.run([R_new, loss, R_weights],feed_dict={X: X_test, D: X_train.T})
		if l > prev_error:
			break
		prev_error = l	
		print("Representation finding loss (MSE) : %.4f" % (l))

	return R_train.T



if __name__ == "__main__":
	parser.init()

	average = 0.0
	for i in range(1,6):
		(X_train, Y_train), (X_test, Y_test) = parser.get_data_split(i)
		# X_train = X_train[:5000]
		# Y_train = Y_train[:5000]
		# X_test = X_test[:1000]
		# Y_test = Y_test[:1000]
		features = find_reprs(X_test.astype("float"), X_train.astype("float"), 0.01, 100)
		indices = np.argmax(features, axis=1)
		Y_predict = []
		for j in indices:
			Y_predict.append(Y_train[j])
		Y_predict = np.array(Y_predict)
		score = np.sum(1*(np.argmax(Y_predict, axis=1) == np.argmax(Y_test, axis=1)))
		score /= float(1000)
		print score
		average += score

	print average / float(5)
