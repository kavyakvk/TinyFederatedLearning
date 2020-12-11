#include <iostream>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>		/* exp */
#include "FCLayer.h" 
#include "NeuralNetwork.cpp"
// #include "FL_model_weights.h"
#include <cstring>
#include <vector>
using namespace std;

int main(void){
	const int input_size = 256;
	const int output_size = 2;
	const double quant_scale = 0.04379776492714882;
	const int quant_zero_point = -128;
	const int fl_devices = 1;
	const int batch_size = 5;
	const int local_epochs = 3;
	const bool real_world = false; // change this if want to use Arducam
	static int current_round = 0;
	int numChars = input_size * batch_size * (sizeof(double) / sizeof(char));

	int batch = 0;

	cout << "creating devices";

	std::vector<FCLayer> devices;
	for(int i = 0; i < fl_devices; i++)
	{
		devices.push_back(FCLayer(input_size, output_size, quant_scale, quant_zero_point, batch_size, true));
 	}

 	FCLayer* NNmodel = &(devices[batch]);
	//save embeddings and ground truths to an array
	cout << "Splitting string for embedding";
	double embeddings_arr[numChars];
	int embedding_index = 0;
	for(int i = 0; i < batch_size*input_size*output_size; i++){
		embeddings_arr[embedding_index] = (double)rand()/(RAND_MAX);
    	embedding_index += 1;
    }

    cout << "finished Splitting string for embedding";

    int gt_arr[batch_size * output_size];
    int gt_index = 0;
    for(int i = 0; i < output_size*batch_size; i++){
        gt_arr[gt_index] = rand() % 10+ 1;
        gt_index += 1;
    }
    cout << "Splitting string for gt";
    cout << "finished Splitting string for gt";

	double **input_data = new double*[NNmodel->batch_size];
 	int **ground_truth = new int*[NNmodel->batch_size];
 	for(int b = 0; b < NNmodel->batch_size; b++) {
		input_data[b] = new double[NNmodel->input_size];
		ground_truth[b] = new int[NNmodel->output_size];
	
	    for(int i = 0; i < NNmodel->input_size; i++){
			input_data[b][i] = embeddings_arr[b*input_size + i]; // INPUT THE VALUE OF THE BTH EMBEDDING AT INDEX I 
	    }
	    for (int j = 0; j < NNmodel->output_size; j++){
			ground_truth[b][j] = gt_arr[b*output_size + j]; //INPUT THE VALUE OF THE BTH GROUND TRUTH AT INDEX I
	    }
	}

	cout << "finished loading data into input data and ground truth";

    FL_round_simulation(input_data, ground_truth, local_epochs, 0.01, &devices[batch], true);

    for(int b = 0; b < batch_size; b++) {
    	delete [] input_data[b];
		delete [] ground_truth[b];
	}
}