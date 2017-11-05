import parser

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adadelta


if __name__ == "__main__":
	parser.init()

	average = 0.0
	for i in range(1,6):
		(X_train, Y_train), (X_test, Y_test) = parser.get_data_split(i)
		model = Sequential()
		model.add(Dense(10, input_shape=(X_train.shape[1],)))
		model.add(Activation('sigmoid'))
		model.add(Dense(5))
		model.add(Activation('softmax'))
		model.layers[0].trainable = False
		model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])
		model.fit(X_train, Y_train, validation_split=0.2, verbose=0)
		score = model.evaluate(X_test, Y_test, verbose=0)[1]
		print score
		average += score

	print average / float(5)
