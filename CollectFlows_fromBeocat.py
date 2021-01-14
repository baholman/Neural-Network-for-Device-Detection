#!/bin/python
import sys

# Check that the python version being used is Python3
major_python_version = sys.version_info[0]
if major_python_version != 3:
	print("ERROR: You need to use Python 3 to run this program")
	exit(1)

import os
from utils.PcapParserHelper import PcapParserHelper
import pcapy as p
from scapy.all import rdpcap, Ether, ARP, IP, TCP, UDP, ICMP, DNS, Raw
import re
import json
import sys
import csv
import signal
import functools
from contextlib import contextmanager
from multiprocessing import Process, Pool

device_identifiers = []
CONVERSATION_THRESHOLD = 5
CORES = 4

experiment_dir = ""
files_already_closed = False

def getDeviceFileName(packet):
	found = False

	# Get identifiying attributes
	src_mac = "Unknown"
	src_ip = "Unknown"

	if packet.haslayer(Ether):
		src_mac = str(packet[Ether].src)

	if packet.haslayer(IP):
		src_ip = str(packet[IP].src)

	# Detemine if the device already exists
	if len(device_identifiers) > 0:
		for device in device_identifiers:
			if device['Ethernet_Source_MAC'] == src_mac:
				return device["name"]

	# Print packet that is not known
	if src_mac == 'Unknown' and src_ip == 'Unknown':
		print(packet)

	# Add new device to list
	new_device_name = "device_" + str(len(device_identifiers))
	device_identifiers.append({
		"name": new_device_name,
		"Ethernet_Source_MAC": src_mac,
		"IP_Source_Address": src_ip
	})

	return new_device_name

def getFlowFilePath(packet, flow_json_dir):
	# Get packet information for filename
	src_mac = ""
	dest_mac = ""
	if packet.haslayer(Ether):
		src_mac = str(packet[Ether].src)
		dest_mac = str(packet[Ether].dst)

	if src_mac == "" and dest_mac == "":
		return "none", True

	# Get the device names for the source and destination devices
	src_device_name = ""
	dest_device_name = ""
	for device_ids in device_identifiers:
		if device_ids['Ethernet_Source_MAC'] == src_mac:
			src_device_name = device_ids['name']
		elif device_ids['Ethernet_Source_MAC'] == dest_mac:
			dest_device_name = device_ids['name']

	# Add a new device for the source device if it couldn't be found
	if src_device_name == "":
		src_ip = ""
		if packet.haslayer(IP):
			src_ip = str(packet[IP].src)

		src_device_name = "device_" + str(len(device_identifiers))
		device_identifiers.append({
			"name": src_device_name,
			"Ethernet_Source_MAC": src_mac,
			"IP_Source_Address": src_ip
		})

	# Add a new device for the destination device if it couldn't be found
	if dest_device_name == "":
		dest_ip = ""
		if packet.haslayer(IP):
			dest_ip = str(packet[IP].dst)

		dest_device_name = "device_" + str(len(device_identifiers))
		device_identifiers.append({
			"name": dest_device_name,
			"Ethernet_Source_MAC": dest_mac,
			"IP_Source_Address": dest_ip
		})

	print('Source mac: ' + src_mac)
	print('Source device: ' + src_device_name)
	print('Destination mac: ' + dest_mac)
	print('Destination device: ' + dest_device_name)

	# Check if the device exists in either device directory
	src_device_flow_dir = os.path.join(flow_json_dir, src_device_name)
	print('Source dir: ' + src_device_flow_dir)
	if os.path.exists(src_device_flow_dir):
		print('Source file exists')
		for filename in os.listdir(src_device_flow_dir):
			if	(filename == src_mac + '-' + dest_mac + '.json') or\
				(filename == dest_mac + '-' + src_mac + '.json'):
				return os.path.join(src_device_flow_dir, filename), False

	dest_device_flow_dir = os.path.join(flow_json_dir, dest_device_name)
	print('Destination dir: ' + dest_device_flow_dir)
	if os.path.exists(dest_device_flow_dir):
		print('Destination file exists')
		for filename in os.listdir(dest_device_flow_dir):
			if	(filename == src_mac + '-' + dest_mac + '.json') or\
				(filename == dest_mac + '-' + src_mac + '.json'):
				return os.path.join(dest_device_flow_dir, filename), False
	'''
	# Check if the directory needs to be created
	if not os.path.exists(src_device_flow_dir) and not os.path.exists(dest_device_flow_dir):
		print('Made new ' + src_device_flow_dir + ' directory')
		# Create the directory for the source device
		os.mkdir(src_device_flow_dir)
	'''
	if not os.path.exists(src_device_flow_dir):
		print('Made new ' + src_device_flow_dir + ' directory')
		# Create the directory for the source device
		os.mkdir(src_device_flow_dir)
	"""
	if not os.path.exists(dest_device_flow_dir):
		print('Made new ' + dest_device_flow_dir + ' directory')
		# Create the directory for the source device
		os.mkdir(dest_device_flow_dir)
	"""
	# Make new file if a file doesn't exist for the flow
	device_flow_path = os.path.join(src_device_flow_dir, src_mac + "-" + dest_mac + ".json")
	device_flow_file = open(device_flow_path, "w")
	device_flow_file.write('{\n')
	device_flow_file.write('	"src_mac": "' + src_mac + '",\n')
	device_flow_file.write('	"src_name": "' + src_device_name + '",\n')
	device_flow_file.write('	"dest_mac": "' + dest_mac + '",\n')
	device_flow_file.write('	"dest_name": "' + dest_device_name + '",\n')
	device_flow_file.write('	"packets": [\n')
	device_flow_file.close()
	return device_flow_path, True

def closeAllFlowFiles(flow_json_dir):
	global files_already_closed

	if files_already_closed == False:
		# Close all the flow files
		for dirname in os.listdir(flow_json_dir):
			device_flow_path = os.path.join(flow_json_dir, dirname)
			if os.path.isdir(device_flow_path):
				for filename in os.listdir(device_flow_path):
					device_flow_file_path = os.path.join(device_flow_path, filename)
					flow_file = open(device_flow_file_path, "a")
					flow_file.write('\n')
					flow_file.write('	]\n')
					flow_file.write('}\n')
					flow_file.close()
			else:
				print("Unexpected file found in " + flow_json_dir + " called " + dirname)

	files_already_closed = True

def download_pcap_files(pcap_file_names, pcap_dir, flow_json_dir):
	# Create the dictionary of packet information split by pcap file
	parser = PcapParserHelper()

	for pcap_file in pcap_file_names:
		print("Currently processing the " + pcap_file + " PCAP file")

		# Set up the input file path
		pcap_path = os.path.join(pcap_dir, pcap_file)

		# Check if file exists
		if not os.path.isfile(pcap_path):
			print("PCAP was not a file")
			continue

		# Get packet info
		for packet in rdpcap(pcap_path):
			# Get path for the file of the device that this packet is associated with
			device_name = getDeviceFileName(packet)

			device_flow_path = getFlowFilePath(packet, flow_json_dir, device_name)
			if device_flow_path != "none":
				device_flow_file = open(device_flow_path, "a")
				device_flow_file.write('		{\n')
				device_flow_file.close()

				# Get packet attributes
				parser.getHeader(packet, device_flow_path, verbose)
				#body = self.__getBody(packet, device_flow_path, verbose)

				device_flow_file = open(device_flow_path, "a")
				device_flow_file.write('		},\n')
				device_flow_file.close()

def signal_handler(sig, frame):
	flow_json_dir = os.path.join(experiment_dir, 'flow_json')
	closeAllFlowFiles(flow_json_dir)
	sys.exit(0)

def main():
	global experiment_dir, device_identifiers

	# Check the number of arguments
	if len(sys.argv) != 3 and len(sys.argv) != 4:
		print('ERROR: Incorrect number of arguments provided')
		print('python3 DeviceDetector.py <experiment_directory> <verbose> <device_names_file>')
		exit(-1)

	# Get the experiment directory
	experiment_dir = sys.argv[1]
	experiment_parent_dir =  ''
	if ((experiment_dir[0] != '/') or (experiment_dir[0] != '~')):
		experiment_parent_dir = os.getcwd()
	else:
		experiment_parent_dir = ''
	if experiment_dir[0] == '.' and len(experiment_dir) == 1:
		experiment_dir = ''
	experiment_dir = os.path.join(experiment_parent_dir, experiment_dir)

	if not os.path.isdir(experiment_dir):
		print('ERROR: The experiment directory provided does not exist')
		return

	# Get the whether the user wants verbose results
	verbose = False
	if sys.argv[2].lower() == "true" or sys.argv[2].lower == "t":
		verbose = True

	# Get the name of the device names file
	if len(sys.argv) == 4:
		device_names = sys.argv[3]
		device_names_parent = ''
		if ((device_names[0] != '/') or (device_names[0] != '~')):
			device_names_parent = os.getcwd()
		else:
			device_names_parent = ''
		device_names_path = os.path.join(device_names_parent, device_names)

		if not os.path.isfile(device_names_path):
			print('ERROR: The device names file provided does not exist')
			return

		device_names_file = open(device_names_path, "r")
		device_names_json = json.loads(device_names_file.read())

		for device_name in device_names_json:
			device = device_names_json[device_name]
			device_identifiers.append({
				"name": device_name,
				"Ethernet_Source_MAC": device["Source MAC Address"],
				"IP_Source_Address": device["IP Address"]
			})

	print("Processing the PCAP files")

	# Check if the content JSON files have already been created
	flow_json_dir =  os.path.join(experiment_dir, 'flow_json')
	if os.path.isdir(flow_json_dir):
		print('The pcap files from this experiment have already been converted to flow JSON files')
	else:
		print('The pcap files for this experiment are being converted to flow JSON files')
		# Make the content_json directory
		os.mkdir(flow_json_dir)

		# Handle Ctrl+C event
		signal.signal(signal.SIGINT, signal_handler)

		# Get the directory for the pcap files
		pcap_dir = os.path.join(experiment_dir, 'pcaps')
		if not os.path.isdir(pcap_dir):
			print('ERROR: The pcap directory provided does not exist')

		# Create the dictionary of packet information split by pcap file
		parser = PcapParserHelper()

		# Split up available PCAPs based on number of cores
		pcap_dir = os.path.join(experiment_dir, "pcaps")
		# Create the dictionary of packet information split by pcap file
		parser = PcapParserHelper()

		for pcap_file in os.listdir(pcap_dir):
			print("Currently processing the " + pcap_file + " PCAP file")

			# Set up the input file path
			pcap_path = os.path.join(pcap_dir, pcap_file)

			# Check if file exists
			if not os.path.isfile(pcap_path):
				print("PCAP was not a file")
				continue

			# Get packet info
			for packet in rdpcap(pcap_path):
				print('new packet')
				# Get path for the file of the device that this packet is associated with
				#device_name = getDeviceFileName(packet)
				#print('device: ' + device_name)

				device_flow_path, is_new_file = getFlowFilePath(packet, flow_json_dir)
				print('Device flow path: ' + device_flow_path)
				print('Is new file? ' + str(is_new_file))

				if device_flow_path != "none":
					device_flow_file = open(device_flow_path, "a")
					if is_new_file:
						device_flow_file.write('		{\n')
					else:
						device_flow_file.write(',\n')
						device_flow_file.write('		{\n')

					device_flow_file.close()

					# Get packet attributes
					parser.getHeader(packet, device_flow_path, verbose)
					#body = self.__getBody(packet, device_flow_path, verbose)

					device_flow_file = open(device_flow_path, "a")

					device_flow_file.write('		}')

					device_flow_file.close()


			print("Finished processing the " + pcap_file + " PCAP file")

		# Close all the flow files
		closeAllFlowFiles(flow_json_dir)

		print(device_identifiers)

		print("Finished processing all files in this run")
			
main()
