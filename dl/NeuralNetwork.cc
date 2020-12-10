// classes example
#include <iostream>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>		/* exp */
#include "NeuralNetwork.h" 
using namespace std;

//TODO: PUT THIS SOMEWHERE srand (time(NULL));

FCLayer::FCLayer (int input_sz, int output_sz, 
					double quant_scale, double quant_zero_point,
					int batch) {
	input_size = input_size;
	output_size = output_size;

	double *bias = new double[output_size];
	for(int i = 0; i < output_size; i++) {
		bias[i] = (double)rand()/(RAND_MAX+1);
	}

	double **weights = new double*[input_size];
	for(int i = 0; i < input_size; ++i) {
	    weights[i] = new double[output_size];
	    for(int j = 0; i < output_size; j++){
			weights[i][j] = (double)rand()/(RAND_MAX+1);
	    }
	}
	int batch_size = batch;
}

void FCLayer::set_weights(double **new_weights){
	for(int i = 0; i < input_size; ++i) {
		for(int j = 0; i < output_size; j++){
			weights[i][j] = new_weights[i][j];
		}
	}
}

double softmax(double *input_data, double *output){
	int classes = sizeof(*output);
	double sum = 0;
	for(int i = 0; i < classes; i++){
		output[i] = exp(input_data[i]);
		sum += output[i];
	}
	for(int i = 0; i < classes; i++){
		output[i] = output[i]/sum;
	}
	return output;
}

void dequantize(int *input_data, double *input_float){
	//use double input [input_size]; when calling
	//input = (input - zero_point) * scale
	for(int i = 0; i < input_size; i++){
		input_float[j] = (input_data[i]-quant_zero_point)*quant_scale;
	}
}


void FCLayer::forward (double **input_float, double **output) {
	/*
	https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
	*/

	//self.output = np.dot(self.input, self.weights) + self.bias
	for(int b = 0; b < batch_size; b++){
		for(int j = 0; j < output_size; j++){
			output[b][j] = 0.0;
			for(int i = 0; i < input_size; i++){
				output[b][j] = input_float[b][i*input_size+j]*weights[i*input_size+j];
			}
			output[b][j] += bias[j];
		}

		//softmax to get probabilities
		softmax(output[b], output_size, output[b]);
	}
}

void cross_entropy_prime(double *pred, double *real, double *result, int batch_size){
	// pred and real are a [# classes] sized array 
	int classes = sizeof(*pred)
	
	for(int b = 0; b < batch_size; b++){
		for(int i = 0; i < classes; i++){
			result[i] = (pred[i]-real[i])/batch_size;
		}
	}
}

void mse_prime(double **pred, double **real, double *result, int batch_size){
	int classes = sizeof(*pred)

	for(int b = 0; b < batch_size; b++){
		for(int i = 0; i < classes; i++){
			result[i] = (pred[i]-real[i])/batch_size;
		}
	}

}

void FCLayer::backward (double **output, double **ground_truth, 
						double **input_error, double** input_float,
						double learning_rate) {
	/*
	https://towardsdatascience.com/neural-networks-from-scratch-easy-vs-hard-b26ddc2e89c7
	*/

	double **output_error = new int*[batch_size];
	for(int b = 0; b < batch_size; ++i) {
	    output_error[b] = new double[output_size];
	}

	cross_entropy_prime(output, ground_truth, output_error, batch_size);

	for(int b = 0; b < batch_size; b++){
		//input_error = np.dot(output_error, self.weights.T)
		// batch_sizexinput_size = batch_sizexoutput_size dot output_sizexinput_size
	    for(int i = 0; i < input_size; ++i) {
	    	input_error[b][i] = 0.0;
		    for(int j = 0; i < output_size; j++){
				input_error[b][i] += output_error[b][j]*weights[i][j];
		    }
		}
	}

	//weights_error = np.dot(self.input.T, output_error)
	double weights_error[input_size][output_size];
	for(int i = 0; i < input_size; ++i) {
		// input_sizexoutput_size = input_sizexbatch_size dot batch_sizexoutput_size
	    weights_error[i] = 0;
		for(int j = 0; i < output_size; j++){
			for(int b = 0; b < batch_size; b++){
				weights_error[i][j] += input_float[b][i]*output_error[b][j];
		    }
		}
	}

	//self.weights -= learning_rate * weights_error
	for(int i = 0; i < input_size; ++i) {
		for(int j = 0; i < output_size; j++){
			weights[i][j] -= learning_rate*weights_error[i][j]/batch_size;
		}
	}
	//self.bias -= learning_rate * output_error
	for(int j = 0; i < output_size; j++){
		for(int b = 0; b < batch_size; ++i) {
			bias[j] -= learning_rate*output_error[b][j]
		}
	}

	for(int b = 0; b < batch_size; b++){
	    delete [] output_error[b];
	}
}

FCLayer::~FCLayer() { 
	for(int i = 0; i < input_size; ++i) {
		delete [] weights[i];
	}

	delete[] bias; 
}

void FL_round(int **input_data, int **ground_truth, int local_episodes, FCLayer *model){

	double **input_float = new double*[model->batch_size];
	double **output = new double*[model->batch_size];
	double **input_error = new double*[model->batch_size];

	for(int b = 0; b < model->batch_size; ++i) {
		input_float[b] = new double*[model->input_size];
		output[b] = new double*[model->output_size];
		input_error[b] = new double*[model->input_size];
	}

	for(int b = 0; b < model->batch_size; ++i) {
		dequantize(input_data[b], input_float[b]);
	}

	for(int epi = 0; epi <= local_episodes; epi++){
		//forward
		model->forward(input_float, output)

		//calculate and print error

		//backward
		model->backward(output, ground_truth, input_error, input_float, learning_rate);
		
		//update learning rate
		//reset input_error and output in forward and backward 
	}

	for(int b = 0; b < model->batch_size; b++) {
		delete [] input_float[b];
		delete [] input_error[b];
		delete [] output[b];
	}
}
