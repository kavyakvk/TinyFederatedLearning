import csv
import pickle

file = open('simulation-results/local_episodes_experiment.txt', 'r') 

trial_counter = -1
param_fl_devices = 0
param_local_episodes = 0
param_batch_size = 0
total_epochs = 0
num_devices = 2

results_local_episodes = {} #results[value of exp_parameter][trial][epoch][device][train accuracy, val accuracy]
with open('simulation-results/results_local_episodes.csv', 'w', newline='') as f:
	writer = csv.writer(f)
	writer.writerows(["local_episodes", "trial", "epoch", "average_device_train_accuracy", "average_device_val_accuracy"])

	for line in file:
		line = line.strip()
		if("===" in line):
			print("starting new group") 
			trial_counter = -1
		elif("START" in line):
			spl = line.split(" ")
			param_fl_devices = int(spl[2])
			param_local_episodes = int(spl[4])
			param_batch_size = int(spl[6])
			total_epochs = 8000//(param_batch_size*param_fl_devices);
			num_devices = param_fl_devices

			results_local_episodes[param_local_episodes] = [[[[0,0] for d in range(param_fl_devices)] for j in range(total_epochs)] for i in range(20)]  #CHANGE to correct param here

			print("params for new group", param_fl_devices, param_local_episodes, param_batch_size, total_epochs)
		
		elif("TRIAL" in line and "END" not in line):
			trial_counter += 1
			assert(trial_counter == int(line.split(" ")[1]))
			print("\ttrial ", trial_counter)
		elif("acc" in line):
			spl = line.split(" ")
			#print(spl)
			epoch = int(spl[1])
			device = int(spl[3])
			train = float(spl[7])
			val = float(spl[10])

			print(train, val)
			results_local_episodes[param_local_episodes][trial_counter][epoch][device][0] = train #CHANGE to correct param here
			results_local_episodes[param_local_episodes][trial_counter][epoch][device][1] = val #CHANGE to correct param here

			if(device == num_devices-1):
				train_score = sum([results_local_episodes[param_local_episodes][trial_counter][epoch][device][0] for device in range(num_devices)])
				val_score = sum([results_local_episodes[param_local_episodes][trial_counter][epoch][device][1] for device in range(num_devices)])

				output = [param_local_episodes, trial_counter, epoch, train_score/num_devices, val_score/num_devices]
				writer.writerows(map(lambda x: [x], output))

	pickle.dump(results_local_episodes, open( "simulation-results/results_local_episodes.p", "wb"))

