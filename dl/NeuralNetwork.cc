// classes example
#include <iostream>
#include "NeuralNetwork.h" 
using namespace std;


FCLayer::FCLayer (int input_size, int output_size) {
  input = input_size;
  output = output_size;
}

void FCLayer::forward (int input_data) {
}

void FCLayer::backward (float error, float learning_rate) {
}