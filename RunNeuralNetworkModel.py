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
import pandas as pd # Data prep library
import numpy as np # Array and matrix library
import csv # Read CSV labels
import json
import os

def main():
	## Read Data ##
	print("Read test")
	path = "test_conv"
	path = os.path.join(os.getcwd(),path)
	X_devices = os.listdir(path)
	X_files = {}
	file_warnings = {}
	device_labels = {}
	for d_path in X_devices:
		d_files = os.listdir(os.path.join(path,d_path))
		X_files[d_path] = d_files
		file_warnings[d_path] = 0
		device_labels[d_path] = {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0}

	maxDist_path = "max_distance.json"
	maxDist_path = os.path.join(os.getcwd(),maxDist_path)
	max_distance = {}
	with open(maxDist_path, 'r') as f:
		max_distance = json.load(f)

	print("Seperate")
	## Seperate Data ##
	device = []
	X = []
	y_output = []

	for key,vals in X_files.items():
		for val in vals:
			f = open(os.path.join(path,key,val))
			conv_file = csv.reader(f, delimiter=',')
			conv = []
			for row in conv_file:
				if len(row) > 0:
					tL = []
					tL.append(float(row[0]))
					tL.append(float(row[1]))
					conv.append(tL)
			X.append(conv)
			device.append(key)
			f.close()

	print("Restore model and run")
	tf.compat.v1.disable_eager_execution()
	## Run Neural Network from saved model ##
	tf.compat.v1.reset_default_graph()
	new_saver = tf.compat.v1.train.import_meta_graph('my_test_model.meta')
	with tf.compat.v1.Session() as sess:
		# Restore model
		new_saver.restore(sess, tf.train.latest_checkpoint('./'))
		graph = tf.compat.v1.get_default_graph()
		# Restore placeholders
		xplaceholder = graph.get_tensor_by_name("xplaceholder:0")
		yplaceholder = graph.get_tensor_by_name("yplaceholder:0")
		#acc = graph.get_tensor_by_name("accuracy")
		#cost = graph.get_tensor_by_name("cost")
		pred = graph.get_tensor_by_name("pred:0")
		print(pred)
		
		for i in range(len(X)):
			conv = np.array(X[i])
			y = [0,0,0,0,0,0,0,0,0]
			#pred = sess.run(pred, feed_dict={xplaceholder: conv})
			max_index, warn = test_input(sess, pred, conv, xplaceholder, yplaceholder, y, max_distance)
			y_output.append(y)
			device_labels[device[i]][str(max_index)] += 1
			if warn:
				file_warnings[device[i]] += 1

	print("Labels for each device: ")
	print(json.dumps(device_labels, sort_keys=True))
	with open("test_device_labels.json", "w") as f:
		json.dump(device_labels, f)

	print("Warnings for each device: ")
	print(json.dumps(file_warnings, sort_keys=True))
	with open("file_warnings.json", "w") as f:
		json.dump(file_warnings, f)


def test_input(sess, pred, conv, xplaceholder, yplaceholder, y, max_distance):
	prediction = pred.eval({xplaceholder: conv}, session=sess)
	#pred = tf.nn.softmax(logit, name="pred").eval({xplaceholder: np.array(c), yplaceholder: np.array(y_test)}, session=sess)
	#pred = tf.round(tf.nn.softmax(logit)).eval({xplaceholder: np.array(c), yplaceholder: np.array(y_test)})
	last_pred = np.array(prediction[len(conv)-1])
	max_val = np.argmax(last_pred)
	y[max_val] += 1
	# Calc distance and compare to max
	dist = np.linalg.norm(np.array(y) - last_pred)
	warn = False
	if dist > max_distance[str(max_val)]:
		warn = True

	return max_val, warn

main()