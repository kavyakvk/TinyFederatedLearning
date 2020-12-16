# Python script to run Federated Learning


# 1. Load pickle file embeddings
# 2. Load initial weight
# 3. Communicate with arduino


import pickle
import serial
import syslog
import time


def image_data_to_string(single_image_data):
	# Processes a single image
	# Returns embedding, ground truth as strings 
	embedding = single_image_data['embedding'].tolist()[0]
	embedding_str = ",".join(map(str, embedding))
	ground_truth_str = ",".join(map(str, single_image_data['ground_truth']))
	return embedding_str, ground_truth_str

def generate_batched_data(image_data, batched_num=5):
	# Returns a list of batched data, where each element is an array of two strings (binary encoded)

	batched_data_lst = []
	image_counter = 0
	while image_counter < len(image_data) - batched_num:
		single_batch_lst = ['','']
		for i in range(batched_num):
			embedding, ground_truth = image_data_to_string(image_data[image_counter])
			image_counter += 1
			if i == 0:
				single_batch_lst[0] = single_batch_lst[0] + embedding
				single_batch_lst[1] = single_batch_lst[1] + ground_truth
			else:
				single_batch_lst[0] = single_batch_lst[0] + ';' + embedding
				single_batch_lst[1] = single_batch_lst[1] + ';' + ground_truth
		single_batch_lst[0] = (single_batch_lst[0] + '\n').encode()
		single_batch_lst[1] = (single_batch_lst[1] + '\n').encode()
		batched_data_lst.append(single_batch_lst)
	return batched_data_lst


def main():
	# Data
	masked_data = pickle.load(open('../../dl/pickle_masked_processed_color_1209.p', 'rb'))
	unmasked_data = pickle.load(open('../../dl/pickle_unmasked_processed_color_1209.p', 'rb'))
	# train_masked = masked_data[:5]
	# train_unmasked = unmasked_data[:5]
	masked_batched = generate_batched_data(masked_data, batched_num=5)
	# print(masked_batched[0])

	# Initial Weights
	init_weights = pickle.load(open('../../dl/pickle_initial_model_weights.p', 'rb'))
	init_bias = [0.08060145, -0.08060154];          # bias for tf (initial)
	init_weights.extend(init_bias)                  # extend bias to end of 1d weights array
	init_weights_str = (",".join(map(str, init_weights)) + "|").encode()
	len_weights_str = len(init_weights_str)
	weights_iters = len_weights_str // 255
	# print(len_weights_str)


	# Setup connection to Arduino
	port = '/dev/cu.usbmodem142301' # change this to what the Arduino Port is
	ard = serial.Serial(port,9600,timeout=10)
	time.sleep(3) # wait for Arduino

	# For one round
	i = 0
	while (i < 1):
		print('Starting a new round')

		# Serial write section
		ard.flush()

		# # Get embedding and ground truth
		# embeddings, ground_truths = masked_batched[0]

		# # Send embedding
		# ard.write(embeddings)
		# print("Mac: sent embeddings")
		# time.sleep(1)

		# # Serial read section
		# msg = ard.read(ard.inWaiting()) # read all characters in buffer
		# print(f"Mac: received {msg}")
		# # print (f"Mac: received {msg.decode('utf-8')}")

		# # Send ground truth
		# ard.write(ground_truths)
		# print("Mac: sent ground truths")
		# time.sleep(1)

		# Read Arduino's response
		# msg = ard.read(ard.inWaiting()) # read all characters in buffer
		# print(f"Mac: received {msg}")

		# _________SENDING WEIGHTS___________
		for j in range(weights_iters):
			# Serial write section
			ard.flush()
			# print("Mac: sent weights and bias")
			ard.write(init_weights_str[j*255:(j+1)*255])

			time.sleep(1)

			# Read Arduino's response
			msg = ard.read(ard.inWaiting()) # read all characters in buffer
			print(f"Arduino received {msg.decode('utf-8')} out of {len_weights_str} bytes")

		# Send last remaining chars
		if weights_iters * 255 < len_weights_str:
			# print(f"last remaining: {init_weights_str[weights_iters*255:]}")
			ard.write(init_weights_str[weights_iters*255:])
			# Read Arduino's response
			msg = ard.read(ard.inWaiting()) # read all characters in buffer
			print(f"Arduino received {msg.decode('utf-8')} out of {len_weights_str} bytes")

		# ________GETTING WEIGHTS_________
		output_weights_str = ''
		num_packets = 8
		for j in range(num_packets):
			time.sleep(5)
			# Read Arduino's response
			msg = ard.read(ard.inWaiting()) # read all characters in buffer
			output_weights_str += msg.decode("utf-8")
			print(f"Server: received {len(output_weights_str)} bytes")
			# print(f"Mac: received {output_weights_str}")
		print(output_weights_str)
		# output_weight_lst = list(map(float, output_weights_str.split()))
		# print(output_weight_lst)
		# print(len(output_weight_lst))


		i = i + 1
	else:
		print ("Exiting")
	exit()


if __name__ == "__main__":
	main()