import csv

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

	for line in file1:
		line = line.strip()
		if("===" in line):
			print("starting new group") 
			trial_counter = -1
		elif("START" in line):
			spl = line.split(" ")
			param_fl_devices = int(spl[3])
			param_local_episodes = int(spl[5])
			param_batch_size = int(spl[7])
			total_epochs = 8000/(param_batch_size*param_fl_devices);

			results_local_episodes[param_local_episodes] = [[[[0,0],[0,0]] for j in range(total_epochs)] for i in range(20)]  #CHANGE to correct param here

			print("params for new group", param_fl_devices, param_local_episodes, param_batch_size, total_epochs)
		
		elif("TRIAL" in line):
			trial_counter += 1
			assert(trial_counter == line.split(" ")[1])
			print("\ttrial ", trial_counter)
		elif("acc" in line):
			spl = line.split(" ")
			epoch = int(spl[1])
			device = int(spl[3])
			train = int(spl[5])
			val = int(spl[7])

			results_local_episodes[param_local_episodes][trial_counter][epoch][device][0] = train #CHANGE to correct param here
			results_local_episodes[param_local_episodes][trial_counter][epoch][device][1] = val #CHANGE to correct param here

			writer.writerows([param_local_episodes, trial_counter, epoch, 
				sum([results_local_episodes[param_local_episodes][trial_counter][epoch][device][0] for device in range(num_devices)]), 
				sum([results_local_episodes[param_local_episodes][trial_counter][epoch][device][1] for device in range(num_devices)])])

	pickle.dump(results_local_episodes, open( "simulation-results/results_local_episodes.p", "wb"))

