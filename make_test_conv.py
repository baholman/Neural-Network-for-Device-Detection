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
from utils.PcapParserHelper import PcapParserHelper
import pcapy as p
#from scapy.all import rdpcap, Ether, ARP, IP, TCP, UDP, ICMP, DNS, Raw
import re
import json
import csv
from datetime import datetime, timedelta
import shutil
import numpy as np

device_ids = []
FLOW_SEPERATOR = timedelta(seconds= 60)

def getConversationAttributesForFlow(flow_file_name, device_dir_path, path, device_dir):
	# Get the contents of the flow file
	flow_file_path = os.path.join(device_dir_path, flow_file_name)
	flow_file = open(flow_file_path, "r")
	start_reading_packets = False
	packets = []
	flow_src_ip = ""
	flow_file_json = json.loads(flow_file.read())
	spot_found = False

	# Sort json into packets by timestamp
	packets = sorted(flow_file_json['packets'], key=lambda p: datetime.utcfromtimestamp(float(p['Packet_Timestamp'])))
	
	### Pull out Timestamps and Packet Lengths ###
	#packet_timeBetweenPackets = []	# Array of time between each packet in conversation
	#packet_lengths = []				# Array of packet lengths in a conversation
	conv = []						# Array representing a conversation
	flow = []						# Array of conversations for a flow
	packet_timeAndLength = []
	#packet_TL = []

	## iterate through each packet in file
	last_conv_timestamp = datetime.utcfromtimestamp(0.0)	# Timestamp of where the conversation starts
	last_timestamp = datetime.utcfromtimestamp(0.0)	# Timestamp of last packet
	isFirst = True			# Is it the first packet

	conv_index = 0

	for packet in packets:
		#print("\n\nNEW PACKET")
		# if it's the first packet in the flow
		if packet["Packet_Length"] != 'Unknown':
			if isFirst:
				packet_timeAndLength.append(float(packet["Packet_Length"]))
				packet_timeAndLength.append(0.0)
				#packet_TL.append(packet_timeAndLength)
				conv.append(packet_timeAndLength)
				packet_timeAndLength = []
				# Changes is first packet to zero
				isFirst = False
			# if it's within the flow
			elif (last_conv_timestamp + FLOW_SEPERATOR) >= datetime.utcfromtimestamp(float(packet["Packet_Timestamp"])):
				# Calc time between the last packet and the new one
				tBetween = (datetime.utcfromtimestamp(float(packet["Packet_Timestamp"])) - last_timestamp).total_seconds()
				packet_timeAndLength.append(float(packet["Packet_Length"]))
				packet_timeAndLength.append(tBetween)
				#packet_TL.append(packet_timeAndLength)
				conv.append(packet_timeAndLength)
				packet_timeAndLength = []
			# if it's not in the flow
			else:
				#conv.append(packet_TL)
				# Adds conversation to flow
				conv_array = np.asarray(conv)
				with open(os.path.join(path,str(str(device_dir) + "-" + str(conv_index))), 'wb') as f:
					np.savetxt(f, conv_array, delimiter=",")
				conv_index += 1
				# Resets conversation data
				packet_timeAndLength = []
				#packet_TL = []
				conv = []
				# Adds newest packet to new conversation and resets conversation timestamp
				packet_timeAndLength.append(float(packet["Packet_Length"]))
				packet_timeAndLength.append(0.0)
				#packet_TL.append(packet_timeAndLength)
				conv.append(packet_timeAndLength)
				packet_timeAndLength = []
				last_conv_timestamp = datetime.utcfromtimestamp(float(packet["Packet_Timestamp"]))
			# Resets last timestamp
			last_timestamp = datetime.utcfromtimestamp(float(packet["Packet_Timestamp"]))
	return conv_index

def main():
	# Check the number of arguments
	if len(sys.argv) != 2:
		print('ERROR: Incorrect number of arguments provided')
		print('python3 Get_NN_Input.py <flow_directory>')
		exit(-1)

	# Check if the content JSON files have already been created
	flow_json_dir = 'JSON'
	if not os.path.isdir(flow_json_dir):
		print('ERROR: You need to process the flow files first')

	path = "test_conv"
	if os.path.isdir(os.path.join(os.getcwd(), path)):
		shutil.rmtree(path)
	os.mkdir(path)
	os.mkdir(os.path.join(path, "thermostat"))

	device_labels = {}
	labels_path = 'device_labels.csv'
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

	device_withNumConvs = {}

	for device_dir in os.listdir(flow_json_dir):
		device_dir_path = os.path.join(flow_json_dir, device_dir)
		if os.path.isdir(device_dir_path):
			#device_withNumConvs.append(device_dir)
			device_ids.append(str(device_dir))
			class_num = device_labels[str(device_dir)]
			if class_num != -1:
				class_folder = str("class_" + str(class_num))
				path = os.path.join("json_conv", class_folder)
				# Get the conversation for the flow
				for flow_file_name in os.listdir(device_dir_path):
					#print(device_dir_path)
					#print(flow_file_name)
					conv_num = getConversationAttributesForFlow(flow_file_name, device_dir_path, path, device_dir)
					device_withNumConvs[device_dir] = conv_num
	

main()