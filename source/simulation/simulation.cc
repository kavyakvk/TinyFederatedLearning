#include <iostream>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>		/* exp */
#include <string>     // std::string, std::stoi
#include "FCLayer.h" 
#include "NeuralNetwork.cpp"
#include <cstring>
#include <vector>
#include <filesystem>
#include <fstream>
#include <string>
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream

using namespace std;

int main_sim(){
	srand (time(NULL));
	//int argc, char** argv
	int input_size = 1280;
	int output_size = 2;
	double quant_scale = 0.04379776492714882;
	int quant_zero_point = -128;
	int fl_devices = 2;//stoi(argv[1]); //number of devices in the simulation
	int local_episodes = 5;//stoi(argv[2]); //number of local epochs each device trains for
	//int data_per_device = 3;//stoi(argv[3]); //number of examples per device
	//int epoch_data_per_device = 9;//stoi(argv[4]); // number of examples during a given epoch
	int batch_size = 20;//stoi(argv[5]); //batch size for the devices, epoch_data_per_device / batch_size = an int
	int epochs = 30;//stoi(argv[6]); // number of total epochs, epoch_data_per_device * epochs <= 1740
	//int data_per_round = data_per_device / epochs;

	assert(8000 >= epochs*fl_devices*batch_size);

	/*
		TESTING VALUES:
		./simulation 1 2 3 9 3 10
	*/

	std::string embedding_data = "embeddings-cat.txt";
	std::string gt_data = "ground_truth-cat.txt";

	//READ DATA
	ifstream data_file(embedding_data);
	ifstream gt_file(gt_data);

	if(!data_file.is_open()) throw runtime_error("Could not open data file");
	if(!gt_file.is_open()) throw runtime_error("Could not open gt file");

	std::string line_data;
	std::string line_gt;

	//CREATE DEVICES
	FCLayer devices[fl_devices];
	for(int d = 0; d < fl_devices; d++){
		devices[d] = FCLayer(input_size, output_size, quant_scale, quant_zero_point, batch_size, false);
	}

	//ALLOCATE MEMORY TO STORE DATA FOR ALL DEVICES
	double ***input_data = new double**[fl_devices];
	int ***ground_truth = new int**[fl_devices];
	for(int d = 0; d < fl_devices; d++){
		input_data[d] = new double*[batch_size];
		ground_truth[d] = new int*[batch_size];
		for(int b = 0; b < batch_size; b++) {
			input_data[d][b] = new double[input_size];
			ground_truth[d][b] = new int[output_size];
			for(int i = 0; i < input_size; i++){
				input_data[d][b][i] = 0.0;
			}
			for (int j = 0; j < output_size; j++){
				ground_truth[d][b][j] = 0;
			}
		}
	}

	//ALLOCATE MEMORY FOR THE SERVER WEIGHTS AND BIAS
	double **server_weights = new double*[input_size];
	double *server_bias = new double[output_size];
	for(int i = 0; i < input_size; i++){
		server_weights[i] = new double[output_size];
		for (int j = 0; j < output_size; j++){
			server_weights[i][j] = 0.0;
			server_bias[j] = 0.0;
		}
	}

	double data_val;
	int gt_val;
	double learning_rate = 0.1;
	double lambda = 0.0001;
	//throw out first line if CSV
	//getline(data_file, line_data);
	//getline(gt_file, line_gt);

	//FOR EPOCHS
	for(int epoch = 0; epoch < epochs; epoch++){
		//TRAIN FOR EACH DEVICE
		for(int d = 0; d < fl_devices; d++){
			FCLayer* NNmodel = &(devices[d]);

			for(int b = 0; b < batch_size; b++) {
				// Extract the line in the file
				getline(data_file, line_data);
				getline(gt_file, line_gt);

				// Create a stringstream from line
				stringstream ss_data(line_data);
				stringstream ss_gt(line_gt);

				for(int i = 0; i < input_size; i++){
					ss_data >> data_val;

					input_data[d][b][i] = data_val;
				}
				ss_gt >> gt_val;
				int sum = 0;
				for (int j = 0; j < output_size; j++){
					if(j == gt_val){
						ground_truth[d][b][j] = 1;
						sum += 1;
					}else{
						ground_truth[d][b][j] = 0;
						sum += 0;
					}
				}
				assert(sum == 1);
			}
			cout << "epoch: " << epoch << " device: " << d << "\n";
			FL_round_simulation(input_data[d], nullptr, ground_truth[d], local_episodes, learning_rate, 
								NNmodel, lambda, false, false, false);
		}
		
		//WEIGHT AVERAGING FOR EACH DEVICE
		for(int i = 0; i < input_size; i++){
			for (int j = 0; j < output_size; j++){
				server_weights[i][j] = 0.0;
				server_bias[j] = 0.0;
			}
		}

		for(int i = 0; i < input_size; i++){
			for (int j = 0; j < output_size; j++){
				for(int d = 0; d < fl_devices; d++){
					server_weights[i][j] += devices[d].weights[i][j]/(fl_devices);
					server_bias[j] += devices[d].bias[j]/(input_size*fl_devices);
				}
			}
		}

		//DEPLOY WEIGHTS TO EACH DEVICE
		for(int d = 0; d < fl_devices; d++){
			devices[d].set_weights_bias(server_weights, server_bias);
		}

		learning_rate = 0.1 * pow(0.96, (epoch*batch_size / 10000));
		
	}

	data_file.close();
	gt_file.close();

	//DE-ALLOCATE MEMORY STORING SERVER WEIGHTS AND BIAS
	for(int i = 0; i < input_size; i++) {
		delete [] server_weights[i];
	}
	delete[] server_weights;
	delete[] server_bias;

	//DE-ALLOCATE MEMORY STORING DEVICES
	for(int d = 0; d < fl_devices; d++){
		cout << fl_devices << "\n";
		devices[d].cleanup();
	}

	//DE-ALLOCATE MEMORY STORING DATA FOR ALL DEVICES
	for(int d = 0; d < fl_devices; d++){
		for(int b = 0; b < batch_size; b++) {
			delete [] input_data[d][b];
			delete [] ground_truth[d][b];
		}
		delete [] input_data[d];
		delete [] ground_truth[d];
	}
	delete [] input_data;
	delete [] ground_truth;
	return 0;
}

//simple_testing_
int simple_testing_main(){
	//int argc, char** argv
	int input_size = 6;
	int output_size = 2;
	double quant_scale = 0.04379776492714882;
	int quant_zero_point = -128;
	int fl_devices = 3;//stoi(argv[1]); //number of devices in the simulation
	int local_episodes = 4;//stoi(argv[2]); //number of local epochs each device trains for
	int batch_size = 10;//stoi(argv[5]); //batch size for the devices, epoch_data_per_device / batch_size = an int
	int epochs = 20;//stoi(argv[6]); // number of total epochs, epoch_data_per_device * epochs <= 1740

	//CREATE DEVICES
	FCLayer devices[fl_devices];
	for(int d = 0; d < fl_devices; d++){
		devices[d] = FCLayer(input_size, output_size, quant_scale, quant_zero_point, batch_size, false);
	}

	//ALLOCATE MEMORY TO STORE DATA FOR ALL DEVICES
	double ***input_data = new double**[fl_devices];
	int ***ground_truth = new int**[fl_devices];
	for(int d = 0; d < fl_devices; d++){
		input_data[d] = new double*[batch_size];
		ground_truth[d] = new int*[batch_size];
		for(int b = 0; b < batch_size; b++) {
			input_data[d][b] = new double[input_size];
			ground_truth[d][b] = new int[output_size];
			for(int i = 0; i < input_size; i++){
				input_data[d][b][i] = 0.0;
			}
			for (int j = 0; j < output_size; j++){
				ground_truth[d][b][j] = 0;
			}
		}
	}

	//ALLOCATE MEMORY FOR THE SERVER WEIGHTS AND BIAS
	double **server_weights = new double*[input_size];
	double *server_bias = new double[output_size];
	for(int i = 0; i < input_size; i++){
		server_weights[i] = new double[output_size];
		for (int j = 0; j < output_size; j++){
			server_weights[i][j] = 0.0;
			server_bias[j] = 0.0;
		}
	}

	double learning_rate = 0.1;
	double lambda = 1;

	//FOR EPOCHS
	for(int epoch = 0; epoch < epochs; epoch++){
		//TRAIN FOR EACH DEVICE
		for(int d = 0; d < fl_devices; d++){
			FCLayer* NNmodel = &(devices[d]);

			for(int b = 0; b < batch_size; b++) {
				// Extract the line in the file
				for(int i = 0; i < input_size; i++){
					//int a=rand()%2;
					input_data[d][b][i] = (double)rand()/(RAND_MAX);//a*1.0;
				}

				int gt = -1;
				int sum = 0;
				if(5*input_data[d][b][0]+2*input_data[d][b][1]-4*input_data[d][b][2]
					-5*input_data[d][b][3]-2*input_data[d][b][4]+4*input_data[d][b][5] >= 0){
					gt = 1;
				}else{
					gt = 0;
				}

				for (int j = 0; j < output_size; j++){
					if(j == gt){
						ground_truth[d][b][j] = 1;
						sum += 1;
					}
					else{
						ground_truth[d][b][j] = 0;
						sum += 0;
					}
				}
				assert(sum == 1);
			}
			cout << "epoch: " << epoch << " device: " << d << "\n";
			FL_round_simulation(input_data[d], nullptr, ground_truth[d], local_episodes, learning_rate, 
				NNmodel, lambda, false, false, false);
		}
		//WEIGHT AVERAGING FOR EACH DEVICE
		for(int i = 0; i < input_size; i++){
			for (int j = 0; j < output_size; j++){
				server_weights[i][j] = 0.0;
				server_bias[j] = 0.0;
			}
		}


		for(int i = 0; i < input_size; i++){
			for (int j = 0; j < output_size; j++){
				for(int d = 0; d < fl_devices; d++){
					server_weights[i][j] += devices[d].weights[i][j]/fl_devices;
					server_bias[j] += devices[d].bias[j]/(input_size*fl_devices);
				}
			}
		}

		//DEPLOY WEIGHTS TO EACH DEVICE
		for(int d = 0; d < fl_devices; d++){
			devices[d].set_weights_bias(server_weights, server_bias);
		}
	}

	double **output = new double*[batch_size];
	for(int b = 0; b < batch_size; b++) {
		output[b] = new double[output_size];
		for(int j = 0; j < output_size; j++){
			output[b][j] = 0.0;
		}
	}

	//DE-ALLOCATE MEMORY STORING SERVER WEIGHTS AND BIAS
	for(int i = 0; i < input_size; i++) {
		delete [] server_weights[i];
	}
	delete[] server_weights;
	delete[] server_bias;

	for(int b = 0; b < batch_size; b++) {
		delete [] output[b];
	}
	delete [] output;

	//DE-ALLOCATE MEMORY STORING DEVICES
	for(int d = 0; d < fl_devices; d++){
		cout << fl_devices << "\n";
		devices[d].cleanup();
	}

	//DE-ALLOCATE MEMORY STORING DATA FOR ALL DEVICES
	for(int d = 0; d < fl_devices; d++){
		for(int b = 0; b < batch_size; b++) {
			delete [] input_data[d][b];
			delete [] ground_truth[d][b];
		}
		delete [] input_data[d];
		delete [] ground_truth[d];
	}
	delete [] input_data;
	delete [] ground_truth;
	return 1;
}

void feature_selection(){
	cout << "hi";
}

void extract_features(double *input, double *output){
	cout << "hi";
}

int main(){
	int i = main_sim();
	//int j = simple_testing_main();
}