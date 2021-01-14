# CSV to Single JSON file

import csv
import json
import os
import random
import shutil

def main():
	path = "json_conv"

	class_name = []
	device_class = {}

	new_path = "json_total_conv"
	if os.path.isdir(os.path.join(os.getcwd(), new_path)):
		shutil.rmtree(new_path)
	os.mkdir(new_path)

	for class_dir in os.listdir(path):
		class_name.append(class_dir)
	
	for class_num in class_name:
		f_names = []
		for f in os.listdir(os.path.join(path, class_num)):
			fn = str(str(class_num)+ "/" + str(f))
			f_names.append(fn)
		device_class[class_num] = f_names

	max_perClass = 5000
	max_add = max_perClass
	j = 0
	file_num = 0
	while (True):
		c = []
		JSON = {}
		
		for n in range(j,max_perClass):
			for class_num in device_class:
				if (len(device_class[class_num]) > n):
					c_name = device_class[class_num][n]
					c.append(c_name)
					
		c = __shuffle(c)

		i = 0
		for f in c:
			conv = __getConv(path, f)
			JSON[i] = conv
			i += 1

		with open(str("json_total_conv/total_conv_" + str(file_num) + ".json"), "w") as f:
			json.dump(JSON, f)

		j += max_add
		if (len(device_class["class_0"]) >= (max_perClass + 1000)):
			max_perClass += max_add
		elif (len(device_class["class_0"]) > max_perClass):
			max_perClass = len(device_class["class_0"])
		else:
			return
		file_num += 1


def __getConv(path, f):
	class_dir = f.split("/")[0]
	fn = f.split("/")[1]

	csv_f = open(os.path.join(path, f))
	conv_file = csv.reader(csv_f, delimiter=',')
	f_conv = []
	f_conv.append(class_dir) # class is listed first
	f_conv.append(fn) # file name is listed second
	for row in conv_file:
		if len(row) > 0:
			tL = []					 # New array contains [Length, Time] per pair
			tL.append(float(row[0])) # Length
			tL.append(float(row[1])) # Time
			f_conv.append(tL)
	csv_f.close()

	return f_conv

def __shuffle(c):
	random.shuffle(c)
	return c


main()