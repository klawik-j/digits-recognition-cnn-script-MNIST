from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

def load_dataset():
	(training_set_input, training_set_output), (test_set_input, test_set_output) = mnist.load_data()
	training_set_input = training_set_input.reshape((training_set_input.shape[0], 28, 28, 1))
	training_set_output = to_categorical(training_set_output)
	return training_set_input, training_set_output

def modify_pixels_range(training_set_input):
	training_set_input_norm = training_set_input.astype('float32')
	training_set_input_norm = training_set_input_norm / 255.0
	return training_set_input_norm

def create_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	opt = SGD(learning_rate=0.01, momentum=0.8)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def train_model(test_set_input, test_set_output, n_folds=5):
	results = []
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	for training_ix, test_ix in kfold.split(test_set_input):
		model = create_model()
		trainingX, trainingY, testX, testY = test_set_input[training_ix], test_set_output[training_ix], test_set_input[test_ix], test_set_output[test_ix]
		model.fit(trainingX, trainingY, epochs=8, batch_size=16, validation_data=(testX, testY), verbose=0)
		_, accuracy = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (accuracy * 100.0))
		results.append(accuracy)
	return results

def run_test_harness():
	trainingX, trainingY = load_dataset()
	trainingX = modify_pixels_range(trainingX)
	results = train_model(trainingX, trainingY)

if __name__ == "__main__":
    run_test_harness()