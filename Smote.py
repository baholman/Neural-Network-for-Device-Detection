#!/bin/python
import sys

# Ignore Future warning error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Check that the python version being used is Python3
major_python_version = sys.version_info[0]
if major_python_version != 3:
	print("ERROR: You need to use Python 3 to run this program")
	exit(1)

import os
import json
import csv
import numpy as np # Array and matrix library
import random

def main():
	path = "json_conv"
	path = os.path.join(os.getcwd(), path)

	class_conv_num, no_synth_conv = __getConvNum(path)

	largest_class_num = 0
	for _,val in class_conv_num.items():
		if val > largest_class_num:
			largest_class_num = val

	__fileSelect(path, class_conv_num, largest_class_num, no_synth_conv)

def __fileSelect(path, class_conv_num, largest_class_num, no_synth_conv):
	for class_dir in os.listdir(path):
		synth_num = 0
		if class_conv_num[class_dir] != 0:
			while class_conv_num[class_dir] < largest_class_num:
				#f1 = np.random.choice(os.listdir(os.path.join(path, class_dir)))
				#f2 = np.random.choice(os.listdir(os.path.join(path, class_dir)))
				f1 = np.random.choice(no_synth_conv[class_dir])
				f2 = np.random.choice(no_synth_conv[class_dir])
				synth_conv = __smote(os.path.join(path, class_dir), f1, f2)
				__writeToCSV(os.path.join(path, class_dir), synth_conv, class_conv_num[class_dir])
				synth_num += 1
				class_conv_num[class_dir] += 1
				
def __writeToCSV(path, synth_conv, num):
	conv = np.asarray(synth_conv)
	with open(os.path.join(path, str("synth-" + str(num))), 'wb') as f:
		np.savetxt(f, conv, delimiter=",")

def __smote(path, f1, f2):
	file1 = open(os.path.join(path, f1))
	conv_file = csv.reader(file1, delimiter=',')
	conv_size_f1 = 0
	f1_conv = []
	for row in conv_file:
		if len(row) > 0:
			tL = []
			tL.append(float(row[0]))
			tL.append(float(row[1]))
			conv_size_f1 += 1
			f1_conv.append(tL)
	file1.close()

	file2 = open(os.path.join(path, f2))
	conv_file = csv.reader(file2, delimiter=',')
	conv_size_f2 = 0
	f2_conv = []
	for row in conv_file:
		if len(row) > 0:
			tL = []
			tL.append(float(row[0]))
			tL.append(float(row[1]))
			conv_size_f2 += 1
			f2_conv.append(tL)
	file2.close()

	conv_size = 0
	if conv_size_f1 > conv_size_f2:
		conv_size = conv_size_f2
	else:
		conv_size = conv_size_f1
	
	synthetic_conv = []
	for i in range(conv_size):
		dif_len = f1_conv[i][0] - f2_conv[i][0]
		dif_time = f1_conv[i][1] - f2_conv[i][1]
		gap = random.uniform(0.0,1.0)
		synth_len = f1_conv[i][0] + gap * dif_len
		synth_time = f1_conv[i][1] + gap * dif_time

		synth_tl = [synth_len, synth_time]
		synthetic_conv.append(synth_tl)

	return synthetic_conv



def __getConvNum(path):
	class_conv_num = {}
	no_synth_conv = {}
	for class_dir in os.listdir(path):
		num_files = 0
		d_names = []
		for f in os.listdir(os.path.join(path, class_dir)):
			if (f.startswith("device")):
				d_names.append(f)
			num_files += 1
		no_synth_conv[class_dir] = d_names
		class_conv_num[class_dir] = num_files
	return class_conv_num, no_synth_conv



main()