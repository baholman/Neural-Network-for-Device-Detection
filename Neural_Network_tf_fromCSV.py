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
#import Get_NN_Input_ToCSV
import csv # Read CSV labels
import json
import os
import random

def main():
	## Read Data ##
	print("Getting Neural Network Input")
	#device_ids, device_withConvNum, path = Get_NN_Input_ToCSV.main()

	path = "json_conv"

	## Setup RNN ##
	# Parameter Specification
	n_classes = 9	# Number of output nodes/classification types
	n_units = 18	# Size of hidden state
	print("hidden layer size = ", n_units)
	n_features = 2	# Number of features in the dataset

	# Define Placeholders
	tf.compat.v1.disable_eager_execution()
	xplaceholder= tf.compat.v1.placeholder('float',[None,n_features], name="xplaceholder") # Holds batch of feature data
	yplaceholder = tf.compat.v1.placeholder('float',[n_classes], name="yplaceholder") # Holds batch of label data

	# Setup Rnn
	print("Setting Up Neural Network")
	cost, optimizer, logit, accuracy, pred = setup_neural_network(xplaceholder, yplaceholder, n_features, n_units, n_classes)

	## Get Device Labels ##
	# Seperate Features and Labels
	labels_path = 'device_labels.csv'
	device_labels = {}

	# Open CSV file and finds device labels
	f = open(labels_path)
	labels_file = csv.reader(f, delimiter=',')
	line_count = 0
	for row in labels_file:
		if len(row) > 0 and str(row[0]) != "" and line_count > 0:
			# Add the current entry to labels
			device_labels[str(row[0])] = int(row[3])
		line_count += 1
	f.close()

	saver = tf.compat.v1.train.Saver()
	# Gets data for each device and runs through neural network
	with tf.compat.v1.Session() as sess:
		tf.compat.v1.global_variables_initializer().run()
		tf.compat.v1.local_variables_initializer().run()
		
		files_perClass = 5000
		total_acc = 0
		total_it = 0

		for iters in range(20):
			X_train_files = []
			X_train = []
			y_train = []
			y_train_overall = []
			total_train_files = 0

			if len(os.listdir(os.path.join(os.getcwd(),path,"class_0"))) != 0:
				X_train_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_0")),files_perClass))
				for i in range(0,files_perClass):
					y_train_overall.append([1,0,0,0,0,0,0,0,0])
				total_train_files += files_perClass
			if len(os.listdir(os.path.join(os.getcwd(),path,"class_1"))) != 0:
				X_train_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_1")),files_perClass))
				for i in range(0,files_perClass):
					y_train_overall.append([0,1,0,0,0,0,0,0,0])
				total_train_files += files_perClass
			if len(os.listdir(os.path.join(os.getcwd(),path,"class_2"))) != 0:
				X_train_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_2")),files_perClass))
				for i in range(0,files_perClass):
					y_train_overall.append([0,0,1,0,0,0,0,0,0])
				total_train_files += files_perClass
			if len(os.listdir(os.path.join(os.getcwd(),path,"class_3"))) != 0:
				X_train_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_3")),files_perClass))
				for i in range(0,files_perClass):
					y_train_overall.append([0,0,0,1,0,0,0,0,0])
				total_train_files += files_perClass
			if len(os.listdir(os.path.join(os.getcwd(),path,"class_4"))) != 0:
				X_train_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_4")),files_perClass))
				for i in range(0,files_perClass):
					y_train_overall.append([0,0,0,0,1,0,0,0,0])
				total_train_files += files_perClass
			if len(os.listdir(os.path.join(os.getcwd(),path,"class_5"))) != 0:
				X_train_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_5")),files_perClass))
				for i in range(0,files_perClass):
					y_train_overall.append([0,0,0,0,0,1,0,0,0])
				total_train_files += files_perClass
			if len(os.listdir(os.path.join(os.getcwd(),path,"class_6"))) != 0:
				X_train_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_6")),files_perClass))
				for i in range(0,files_perClass):
					y_train_overall.append([0,0,0,0,0,0,1,0,0])
				total_train_files += files_perClass
			if len(os.listdir(os.path.join(os.getcwd(),path,"class_7"))) != 0:
				X_train_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_7")),files_perClass))
				for i in range(0,files_perClass):
					y_train_overall.append([0,0,0,0,0,0,0,1,0])
				total_train_files += files_perClass
			if len(os.listdir(os.path.join(os.getcwd(),path,"class_8"))) != 0:
				X_train_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_8")),files_perClass))
				for i in range(0,files_perClass):
					y_train_overall.append([0,0,0,0,0,0,0,0,1])
				total_train_files += files_perClass
			
			print("Shuffling Train Data")
			index_shuf = list(range(len(X_train_files)))
			random.shuffle(index_shuf)
			
			for i in index_shuf:
				class_index = np.argmax(np.array(y_train_overall[i]))
				class_name = str("class_" + str(class_index))

				f = open(os.path.join(os.getcwd(), path, class_name, str(X_train_files[i])))
				conv_file = csv.reader(f, delimiter=',')
				conv = []
				for row in conv_file:
					if len(row) > 0:
						tL = []
						tL.append(float(row[0]))
						tL.append(float(row[1]))
						conv.append(tL)
				X_train.append(conv)
				y_train.append(y_train_overall[i])
				f.close()

			total_acc += train_neural_network(sess,X_train,y_train,xplaceholder,yplaceholder,cost,optimizer, accuracy,total_train_files)
			total_it += 1

		Training_Accuracy = total_acc / total_it
		print("Total Training Accuracy", Training_Accuracy)

		saver.save(sess,"my_test_model")

		#True_Number = {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0,"9": 0,"10": 0,"11": 0,"12": 0,"13": 0,"14": 0,"15": 0,"16": 0,"17": 0,"18": 0,"19": 0,"20": 0}
		#Total_Number = {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0,"9": 0,"10": 0,"11": 0,"12": 0,"13": 0,"14": 0,"15": 0,"16": 0,"17": 0,"18": 0,"19": 0,"20": 0}
		True_Number = {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0}
		Total_Number = {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0}
		max_distance = {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0}

		wrong_id = {"class_0": {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0},
					"class_1": {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0},
					"class_2": {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0},
					"class_3": {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0},
					"class_4": {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0},
					"class_5": {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0},
					"class_6": {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0},
					"class_7": {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0},
					"class_8": {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0}}
		
		test_files_perClass = 500

		for iters in range(20):
			X_test_files = []
			X_test = []
			y_test = []
			y_test_overall = []
			test_total_files = 0

			if len(os.listdir(os.path.join(os.getcwd(),path,"class_0"))) != 0:
				X_test_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_0")),test_files_perClass))
				for i in range(0,test_files_perClass):
					y_test_overall.append([1,0,0,0,0,0,0,0,0])
				test_total_files += test_files_perClass
			if len(os.listdir(os.path.join(os.getcwd(),path,"class_1"))) != 0:
				X_test_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_1")),test_files_perClass))
				for i in range(0,test_files_perClass):
					y_test_overall.append([0,1,0,0,0,0,0,0,0])
				test_total_files += test_files_perClass
			if len(os.listdir(os.path.join(os.getcwd(),path,"class_2"))) != 0:
				X_test_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_2")),test_files_perClass))
				for i in range(0,test_files_perClass):
					y_test_overall.append([0,0,1,0,0,0,0,0,0])
				test_total_files += test_files_perClass
			if len(os.listdir(os.path.join(os.getcwd(),path,"class_3"))) != 0:
				X_test_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_3")),test_files_perClass))
				for i in range(0,test_files_perClass):
					y_test_overall.append([0,0,0,1,0,0,0,0,0])
				test_total_files += test_files_perClass
			if len(os.listdir(os.path.join(os.getcwd(),path,"class_4"))) != 0:
				X_test_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_4")),test_files_perClass))
				for i in range(0,test_files_perClass):
					y_test_overall.append([0,0,0,0,1,0,0,0,0])
				test_total_files += test_files_perClass
			if len(os.listdir(os.path.join(os.getcwd(),path,"class_5"))) != 0:
				X_test_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_5")),test_files_perClass))
				for i in range(0,test_files_perClass):
					y_test_overall.append([0,0,0,0,0,1,0,0,0])
				test_total_files += test_files_perClass
			if len(os.listdir(os.path.join(os.getcwd(),path,"class_6"))) != 0:
				X_test_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_6")),test_files_perClass))
				for i in range(0,test_files_perClass):
					y_test_overall.append([0,0,0,0,0,0,1,0,0])
				test_total_files += test_files_perClass
			if len(os.listdir(os.path.join(os.getcwd(),path,"class_7"))) != 0:
				X_test_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_7")),test_files_perClass))
				for i in range(0,test_files_perClass):
					y_test_overall.append([0,0,0,0,0,0,0,1,0])
				test_total_files += test_files_perClass
			if len(os.listdir(os.path.join(os.getcwd(),path,"class_8"))) != 0:
				X_test_files.extend(np.random.choice(os.listdir(os.path.join(os.getcwd(),path,"class_8")),test_files_perClass))
				for i in range(0,test_files_perClass):
					y_test_overall.append([0,0,0,0,0,0,0,0,1])
				test_total_files += test_files_perClass
			
			print("Shuffling Test Data")
			index_shuf = list(range(len(X_test_files)))
			random.shuffle(index_shuf)

			label_dist = {"0":[], "1":[], "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[]}
			
			for i in index_shuf:
				class_index = np.argmax(np.array(y_test_overall[i]))
				class_name = str("class_" + str(class_index))

				f = open(os.path.join(path, class_name, str(X_test_files[i])))
				conv_file = csv.reader(f, delimiter=',')
				conv = []
				for row in conv_file:
					if len(row) > 0:
						tL = []
						tL.append(float(row[0]))
						tL.append(float(row[1]))
						conv.append(tL)
				X_test.append(conv)
				y_test.append(y_test_overall[i])
				f.close()

			label = 0
			for c in X_test:
				y = y_test[label]
				TF, max_index, label_dist = test_neural_network(sess,pred,xplaceholder,yplaceholder,c,y,max_distance, label_dist)
				for i in range(len(y)):
					if y[i] == 1 and TF:
						True_Number[str(i)] += 1
						Total_Number[str(i)] += 1
					elif y[i] == 1:
						Total_Number[str(i)] += 1
						wrong_id[str("class_" + str(i))][str(max_index)] += 1
				label += 1
			
		max_distance = __findMaxDist(label_dist)

		print("Correctly predicted convs: ")
		print(json.dumps(True_Number, sort_keys=True))
		with open("True_Number_Convs.json", "w") as f:
			json.dump(True_Number, f)

		print("Total convs tested: ")
		print(json.dumps(Total_Number, sort_keys=True))
		with open("Total_Number_Convs.json", "w") as f:
			json.dump(True_Number, f)

		print("Max distance for each class: ")
		print(json.dumps(max_distance, sort_keys=True))
		with open("max_distance.json", "w") as f:
			json.dump(max_distance, f)

		print("Incorrect Labels: ")
		print(json.dumps(wrong_id, sort_keys=True))
		with open("Incorrect_Labels.json", "w") as f:
			json.dump(wrong_id, f)

	return

	

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
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name="accuracy")

	pred = tf.nn.softmax(logit, name="pred")

	cost = tf.reduce_mean(tf.compat.v2.nn.softmax_cross_entropy_with_logits(logits=logit, labels=yplaceholder), name="cost")
	optimizer = tf.compat.v1.train.AdamOptimizer().minimize(cost)
	return cost, optimizer, logit, accuracy, pred

def train_neural_network(sess, X_train,y_train,xplaceholder,yplaceholder,cost,optimizer, accuracy, iters):
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

def test_neural_network(sess, pred, xplaceholder, yplaceholder,c, y_test, max_distance, label_dist):
	prediction = pred.eval({xplaceholder: np.array(c), yplaceholder: np.array(y_test)}, session=sess)
	#pred = tf.nn.softmax(logit, name="pred").eval({xplaceholder: np.array(c), yplaceholder: np.array(y_test)}, session=sess)
	#pred = tf.round(tf.nn.softmax(logit)).eval({xplaceholder: np.array(c), yplaceholder: np.array(y_test)})
	last_pred = np.array(prediction[len(c)-1])
	max_val = np.argmax(last_pred)
	label = np.argmax(np.array(y_test))
	correct = (y_test[max_val] == 1)

	dist = np.linalg.norm(np.array(y_test) - last_pred)
	label_dist[str(label)].append(dist)
	"""
	# Calc distance and compare to max
	if correct and last_pred[max_val] >= 0.80: # Accuracy greater than or equal to 80%
		dist = np.linalg.norm(np.array(y_test) - last_pred)
		if max_distance[str(max_val)] < (dist * 1.1):
			max_distance[str(max_val)] = dist * 1.1 # 1.1 to add 10% buffer
	"""

	return correct, max_val, label_dist

def __findMaxDist(label_dist):
	max_dist = {}
	
	for key in label_dist:
		order_dist = label_dist[key]
		if order_dist != []:
			order_dist.sort()
			"""
			#For testing
			ld = np.asarray(order_dist)
			with open(os.path.join(os.getcwd(),str("distance" + "-" + str(key) + ".csv")), 'wb') as f:
				np.savetxt(f, ld, delimiter=",")
			"""
			label_dist[key] = order_dist
			dist_len = len(order_dist)
			"""	
			median = order_dist[int(dist_len*.5)]
			Q1 = order_dist[int(dist_len*.25)]
			Q3 = order_dist[int(dist_len*.75)]
			IQR = Q3 - Q1
			
			maximum = Q3 + 1.5*IQR
			"""
			boundary = int(dist_len*.9) # Boundary set at 90%
			maximum = order_dist[boundary]

			max_dist[key] = maximum
	return max_dist

main()