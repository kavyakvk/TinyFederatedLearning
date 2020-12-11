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
	cout << "starting program \n";

	int input_size = 256;
	int output_size = 2;
	double quant_scale = 0.04379776492714882;
	int quant_zero_point = -128;
	int fl_devices = 1;
	int batch_size = 1;
	int local_epochs = 3;
	bool real_world = false; // change this if want to use Arducam
	int current_round = 0;
	int numChars = input_size * batch_size;

	int batch = 0;

	cout << " creating devices \n";

	//FCLayer hi = FCLayer(input_size, output_size, quant_scale, quant_zero_point, batch_size, true);

	FCLayer devices[fl_devices];

	for(int i = 0; i < fl_devices; i++){
		devices[i] = FCLayer(input_size, output_size, quant_scale, quant_zero_point, batch_size, true);
 	}

	FCLayer* NNmodel = &(devices[batch]);
	//save embeddings and ground truths to an array
	cout << "Splitting string for embedding\n";

	double embeddings_arr[batch_size*input_size];
	int embedding_index = 0;
	for(int i = 0; i < batch_size*input_size; i++){
		embeddings_arr[embedding_index] = 1.1;//(double)rand()/(RAND_MAX);
    	embedding_index += 1;
    }

    cout << "finished Splitting string for embedding\n";

    int gt_arr[batch_size * output_size];
    for(int i = 0; i < output_size*batch_size; i++){
        gt_arr[i] = 5;
    }
    cout << "Splitting string for gt\n";
    cout << "finished Splitting string for gt\n";

	double **input_data = new double*[NNmodel->batch_size];
 	int **ground_truth = new int*[NNmodel->batch_size];
 	cout << "finished allocating input_data and ground_truth \n";
 	for(int b = 0; b < NNmodel->batch_size; b++) {
		input_data[b] = new double[NNmodel->input_size];
		ground_truth[b] = new int[NNmodel->output_size];
		cout << "finished allocating actual arrays for input_data and ground_truth \n";
	    for(int i = 0; i < NNmodel->input_size; i++){
			input_data[b][i] = embeddings_arr[b*input_size + i];
	    }
	    cout << "\n";
	    for (int j = 0; j < NNmodel->output_size; j++){
			ground_truth[b][j] = gt_arr[b*output_size + j];
	    }
	    cout << "\nfinished loop\n";
	}

	cout << "finished loading data into input data and ground truth";

    FL_round_simulation(input_data, ground_truth, local_epochs, 0.01, NNmodel, true);

    NNmodel->cleanup();

    for(int b = 0; b < batch_size; b++) {
    	delete [] input_data[b];
		delete [] ground_truth[b];
	}
}