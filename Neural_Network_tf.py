#!/bin/python

# Blake A Holman
# Neural Network using Tensorflow
# URL: https://blog.goodaudience.com/first-experience-of-building-a-lstm-model-with-tensorflow-e632bde911e1

import sys

# Ignore Future warning error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
# Check that the python version being used is Python3
major_python_version = sys.version_info[0]
if major_python_version != 3:
	print("ERROR: You need to use Python 3 to run this program")
	exit(1)

import tensorflow as tf # Machine Learning framework
import numpy as np # Array and matrix library
#from tensorflow.contrib import rnn
#from tensorflow import keras
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import Get_NN_Input
import csv # Read CSV labels
import json

def main():
	## Read Data ##
	print("Getting Neural Network Input")
	devices, device_ids = Get_NN_Input.main()

	## Setup RNN ##
	# Parameter Specification
	n_classes = 5	# Number of output nodes/classification types
	n_units = 10	# Size of hidden state
	n_features = 2	# Number of features in the dataset

	# Define Placeholders
	tf.compat.v1.disable_eager_execution()
	xplaceholder= tf.compat.v1.placeholder('float',[None,n_features]) # Holds batch of feature data
	yplaceholder = tf.compat.v1.placeholder('float',[n_classes]) # Holds batch of label data

	# Setup Rnn
	print("Setting Up Neural Network")
	cost, optimizer, logit, accuracy = setup_neural_network(xplaceholder, yplaceholder, n_features, n_units, n_classes)

	## Get Device Labels ##
	# Seperate Features and Labels
	labels_path = 'device_labels.csv'
	device_labels = []

	# Open CSV file and finds device labels
	f = open(labels_path)
	labels_file = csv.reader(f, delimiter=',')
	for i in device_ids:
		line_count = 0
		for row in labels_file:
			if line_count > 0 and len(row) > 0:
				# Add the current entry to labels
				if i == row[0]:
					device_labels.append(int(row[3]))
			line_count += 1
		f.seek(0)
	f.close()

	## Seperate Data and Run RNN ##
	# Initialize test data
	X_train = []
	X_test = []
	#y_train = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	#y_test = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	y_train = [0,0,0,0,0]
	y_test = [0,0,0,0,0]
	y_train_overall = []
	y_test_overall = []
	print("Adding Train and Test Data")
	i = 0
	for d in devices:
		for f in d:
			j = 0
			for c in f:
				# Train and Test Spit at 10%
				if device_ids[i] != -1:
					if j != 10:
						X_train.append(c)
						y_train[device_labels[i]] = 1
						y_train_overall.append(y_train)
						j += 1
					else:
						X_test.append(c)
						y_test[device_labels[i]] = 1
						y_test_overall.append(y_test)
						j = 0
					
			# Clears data
			#y_train = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
			#y_test = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
			y_train = [0,0,0,0,0]
			y_test = [0,0,0,0,0]
				
		i += 1

	saver = tf.compat.v1.train.Saver()
	# Gets data for each device and runs through neural network
	with tf.compat.v1.Session() as sess:
		tf.compat.v1.global_variables_initializer().run()
		tf.compat.v1.local_variables_initializer().run()
		i = 0
		print("Shuffling Train Data")
		seed = np.random.randint(0,10000)
		np.random.seed(seed)
		np.random.shuffle(X_train)
		np.random.seed(seed)
		np.random.shuffle(y_train_overall)

		len_train = len(X_train)
		total_acc = 0
		total_it = 0
		training_iters = 10000
		step = 0
		while step < training_iters:
			y = y_train_overall[step:training_iters]
			total_acc += train_neural_network(sess,X_train[step:training_iters],y,xplaceholder,yplaceholder,cost,optimizer,logit,accuracy,(training_iters-step))
			total_it += 1
			step = training_iters
			if (training_iters + 10000) <= len_train:
				training_iters += 10000
			else:
				training_iters = len_train
		Training_Accuracy = total_acc / total_it
		print("Total Training Accuracy", Training_Accuracy)

		saver.save(sess,"my_test_model")

		label = 0
		#True_Number = {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0,"9": 0,"10": 0,"11": 0,"12": 0,"13": 0,"14": 0,"15": 0,"16": 0,"17": 0,"18": 0,"19": 0,"20": 0}
		#Total_Number = {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0,"9": 0,"10": 0,"11": 0,"12": 0,"13": 0,"14": 0,"15": 0,"16": 0,"17": 0,"18": 0,"19": 0,"20": 0}
		True_Number = {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0}
		Total_Number = {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0}
		for c in X_test:
			y = y_test_overall[label]
			TF = test_neural_network(sess,logit,xplaceholder,yplaceholder,c,y)
			for i in range(len(y)):
				if y[i] == 1 and TF:
					True_Number[str(i)] += 1
					Total_Number[str(i)] += 1
				elif y[i] == 1:
					Total_Number[str(i)] += 1
			label += 1
		print(json.dumps(True_Number, sort_keys=True))
		print(json.dumps(Total_Number, sort_keys=True))

	

def recurrent_neural_network_model(xplaceholder, n_features, n_units, n_classes):
    # giving the weights and biases random values
	layer ={ 'weights': tf.Variable(tf.random.normal([n_units, n_classes])),'bias': tf.Variable(tf.random.normal([n_classes]))}

	#x = tf.reshape(xplaceholder, [-1, n_features])
	x = tf.split(xplaceholder, n_features, 1)

    # x is a 2-dimensional Tensor and it is sliced along the dimension 1 (columns),
    # each slice is an element of the sequence given as input to the LSTM layer.
    # creates a LSTM layer and instantiates variables for all gates.
    # rnn_size is the size of your hidden state (both c and h in a LSTM).
	#lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_units)
	lstm_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([tf.compat.v1.nn.rnn_cell.LSTMCell(n_units),tf.compat.v1.nn.rnn_cell.LSTMCell(n_units)])
    # outputs contains the output for each slice of the layer
    # sate contains the final values of the hidden state
	outputs, states = tf.compat.v1.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
	
    # for each element (<tf.Tensor 'split:0' shape=(<batch size>, 1)) of the input sequence.
	output = tf.matmul(outputs[-1], layer['weights']) + layer['bias']
	
	return output

def setup_neural_network(xplaceholder, yplaceholder, n_features, n_units, n_classes):
	logit = recurrent_neural_network_model(xplaceholder, n_features, n_units, n_classes)

	logit = tf.reshape(logit, [-1, n_classes])
	print(logit.shape)

	correct_pred = tf.equal(tf.argmax(logit,1), tf.argmax(yplaceholder))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	cost = tf.reduce_mean(tf.compat.v2.nn.softmax_cross_entropy_with_logits(logits=logit, labels=yplaceholder))
	optimizer = tf.compat.v1.train.AdamOptimizer().minimize(cost)
	return cost, optimizer, logit, accuracy

def train_neural_network(sess, X_train,y_train,xplaceholder,yplaceholder,cost,optimizer,logit, accuracy, iters):
	loss_total = 0
	loss = 0
	i = 0
	batch_loss = 0
	batch_acc = 0
	total_acc = 0
	for i in range(8):
		for c in range(len(X_train)):
			batch_x = np.array(X_train[c])
			batch_y = np.array(y_train[c])
			# runs the computation subgraph necessary for the specified operation.
			_, loss, acc = sess.run([optimizer, cost, accuracy], feed_dict={xplaceholder: batch_x, yplaceholder: batch_y})
			batch_loss += loss
			batch_acc += acc
		loss_total += batch_loss
		total_acc += batch_acc
		batch_acc = 0
		batch_loss = 0
		i += 1
	total_acc_forEpoch = (total_acc/(iters*8))
	print("Total loss for 8 epochs in train: ", loss_total)
	print("Total accuracy for 8 epoch in train: ", total_acc_forEpoch)
	return total_acc_forEpoch

def test_neural_network(sess, logit, xplaceholder, yplaceholder,c, y_test):
	pred = tf.nn.softmax(logit).eval({xplaceholder: np.array(c), yplaceholder: np.array(y_test)})
	#pred = tf.round(tf.nn.softmax(logit)).eval({xplaceholder: np.array(c), yplaceholder: np.array(y_test)})
	for i in range(len(c)):
		max_index = np.argmax(np.array(pred[i]))
		if y_test[max_index] != 1:
			return False
	return True

main()