// classes example
#include <iostream>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>		/* exp */
#include "FCLayer.h" 
// #include "FL_model_weights.h"
#include <cstring>
//#include "Particle.h"
//#include <Arduino.h>
using namespace std;

//TODO: PUT THIS SOMEWHERE srand (time(NULL));

/*
ActivationLayer::ActivationLayer(double **inp){
	double **input_data = inp;
}

ActivationLayer::forward(double **input_float, double **output){
	input_size = sizeof(*input_data[0])
	for(int i = 0; i < input_size; ++i){
		delete [] input_data[i];
	}

	input_data = input_float;

	for(int b = 0; b < batch_size; b++){
		for(int j = 0; j < output_size; j++){
			output[b][j] = tanh(output[b][j]);
		}
	}
}

ActivationLayer::backward(double **output_error, double **input_error, double double learning_rate){
	//the length of input_error is the output_size of the previous FC layer in the stack
	for(int b = 0; b < batch_size; b++){
		double err = 0.0;
		for(int j = 0; j < sizeof(*input_error[0]); j++){
			input_error[b][j] = tanh(input_data);
		}
	}
}

ActivationLayer::~ActivationLayer() { 
	input_size = sizeof(*input_data[0])
	for(int i = 0; i < input_size; ++i){
		delete [] input_data[i];
	}
}
*/



const double initial_model_bias[] = {0.08060145, -0.08060154};
const double initial_model_weights[] = {
		0.07168999,  0.07093393,  0.14018372,  0.14018327,  0.05810054,
        0.05802297, -0.07410979, -0.0738539 , -0.01224348, -0.01215865,
       -0.14877185,  0.2333137 ,  0.10125715,  0.10134993, -0.07413147,
       -0.07439099, -0.06184382, -0.06181126,  0.00083795,  0.00149697,
       -0.03897167, -0.03872154, -0.01879336, -0.01878951, -0.05627485,
       -0.05625216, -0.36613655,  0.40711913,  0.14664188, -0.24305038,
       -0.10275028, -0.1026675 , -0.09283438, -0.09283756, -0.04795968,
       -0.0481106 , -0.4814787 ,  0.4945571 , -0.01384905, -0.00447169,
       -0.00767066, -0.00761715, -0.01137266, -0.01135145,  0.03956768,
        0.03957317, -0.03381831, -0.03419717,  0.09204329,  0.09184879,
        0.03686117,  0.03688825, -0.09403573, -0.09405135,  0.02920679,
        0.02933406, -0.08844789, -0.08853102, -0.0847619 , -0.08311669,
        0.08026171,  0.0796629 , -0.07432654, -0.0744877 ,  0.03903978,
        0.0383551 ,  0.07692228,  0.07682239,  0.00739892,  0.0074377 ,
        0.04621403,  0.04631572,  0.06233774,  0.06238936,  0.06373248,
        0.06386399,  0.03373879,  0.03407782,  0.05577807,  0.05622289,
        0.05466111,  0.0564717 ,  0.05694389,  0.05675327,  0.02709804,
        0.02712604,  0.00361158,  0.00369934,  0.23855841, -0.22338964,
        0.02012629,  0.02012875, -0.06190006, -0.06239133, -0.26886633,
        0.25226992, -0.10019336, -0.10000549, -0.06415121, -0.06406697,
        0.05390483,  0.05398929,  0.09964863,  0.10022175,  0.04989993,
        0.04905739,  0.07506058,  0.07595515,  0.01704206,  0.01707293,
       -0.00619068, -0.01463624, -0.04503923, -0.0451085 , -0.04831591,
       -0.0476345 , -0.1376636 , -0.13779026, -0.11013918, -0.12440069,
        0.13642706,  0.13636684, -0.05228312, -0.05233908,  0.06509004,
        0.06506539, -0.05138277, -0.05140109,  0.05080156,  0.05133656,
       -0.078444  , -0.07835598,  0.02016387,  0.02048565, -0.0376487 ,
       -0.03731735, -0.00939265, -0.00931421, -0.00468205, -0.00492001,
        0.06138162,  0.06145221,  0.03585195,  0.03597203, -0.31984463,
        0.31966817,  0.36544725, -0.38819382, -0.06061538, -0.06130901,
        0.00675081,  0.00679944, -0.05065987, -0.05070452, -0.13896815,
       -0.13904308,  0.09265722,  0.09334209, -0.13315712, -0.13289574,
       -0.07197165, -0.07229659,  0.0745015 ,  0.07469526, -0.05933112,
       -0.05832031, -0.02173107, -0.02179518,  0.10650945,  0.10675649,
       -0.08783317, -0.08772595, -0.05810835, -0.05612781, -0.02444657,
       -0.02465268, -0.0199524 , -0.0198834 ,  0.074536  ,  0.07458106,
        0.10033005,  0.10021254, -0.35598022,  0.39040852,  0.13405837,
        0.13416667,  0.01963953,  0.01966765, -0.03573176, -0.0359267 ,
       -0.07980753, -0.07970313,  0.06001445,  0.05382619, -0.06397508,
       -0.06423512,  0.08237936,  0.08252191,  0.03839105,  0.03844513,
       -0.06521061, -0.06501263,  0.02733659,  0.02721914, -0.06881996,
       -0.06885421, -0.03320974, -0.03319879,  0.04365296,  0.04354954,
       -0.05206276, -0.05208507, -0.01301841, -0.0128294 , -0.00238957,
       -0.00235198, -0.32834497,  0.3306103 , -0.05022464, -0.05011163,
       -0.0806412 , -0.08063363, -0.00679854, -0.01126088,  0.00134615,
        0.00142438, -0.4066513 ,  0.39144376,  0.0265792 ,  0.02659874,
       -0.01470394, -0.01656722,  0.08347063,  0.08214717,  0.31929684,
       -0.23629592, -0.10551224, -0.10551513,  0.03633926,  0.03634805,
       -0.0117904 , -0.00981981,  0.03934224,  0.03955718,  0.10006125,
        0.10013341, -0.04423587, -0.04411171,  0.02447847,  0.024564  ,
       -0.06527217, -0.06519797,  0.06725655,  0.06810225,  0.051027  ,
        0.0509846 ,  0.03925848,  0.03770661, -0.0031315 , -0.00301287,
       -0.00624296, -0.00611177, -0.04380893, -0.04539516, -0.01043836,
       -0.10088602,  0.00072343,  0.00138796,  0.26517537, -0.33121917,
        0.09027502,  0.09012288,  0.04102091,  0.04337964,  0.02024079,
        0.02055687,  0.00917042,  0.00791404, -0.01408286, -0.01440221,
       -0.10255095, -0.10247644,  0.02348975,  0.02334828,  0.2428903 ,
       -0.25765017,  0.02202941,  0.02162507, -0.1118564 , -0.11181158,
       -0.08017021, -0.08021232,  0.11662102,  0.11671803, -0.06997235,
       -0.07077195,  0.08290115,  0.08272505,  0.00068876, -0.00185605,
       -0.09746628, -0.09746519, -0.0154012 , -0.01617656,  0.00282206,
        0.00314476, -0.01569918, -0.01599419, -0.02435168, -0.02437891,
       -0.5355525 ,  0.58406484, -0.11753685, -0.11745376, -0.11570402,
       -0.11562854, -0.05104811, -0.05075606, -0.02305056, -0.02455798,
       -0.04206524, -0.04171735,  0.02685071,  0.02694452,  0.22764187,
       -0.18062383,  0.0708196 ,  0.0707995 , -0.0435198 , -0.04348541,
       -0.03240912, -0.03224351,  0.10352735,  0.10356284,  0.08932678,
        0.08933071, -0.06191229, -0.06127301,  0.12138531,  0.12174109,
       -0.01989888, -0.00743882, -0.04207218, -0.04164417, -0.08312049,
       -0.06642811, -0.08697369, -0.08764228, -0.02516148, -0.02506847,
       -0.02326604, -0.02355509, -0.07983012, -0.0798711 , -0.2728246 ,
        0.3655748 ,  0.0584928 ,  0.05771175,  0.07465177,  0.07435394,
       -0.01281226, -0.01209527,  0.05651   ,  0.05658976,  0.0609    ,
        0.06084421, -0.0533757 , -0.05343005, -0.07783872, -0.07783476,
       -0.01632413, -0.01700388, -0.055774  , -0.05577601, -0.06283636,
       -0.06756648, -0.11047469, -0.11057252,  0.01107963,  0.0109978 ,
       -0.10563431, -0.10584833, -0.07052658, -0.0655302 ,  0.0230823 ,
        0.02347211, -0.11757515, -0.1184411 ,  0.00879552,  0.00880601,
        0.07194074,  0.07197646,  0.03061995,  0.02828274, -0.0032193 ,
       -0.00370012, -0.02649869, -0.02661759, -0.44400933,  0.39838797,
        0.01254125,  0.0125876 ,  0.00623139,  0.00655925,  0.01093041,
        0.01090835, -0.01220039, -0.01209055, -0.03383605, -0.03339904,
        0.07915339,  0.0793725 ,  0.03278508,  0.03267733, -0.05888168,
       -0.05864355, -0.06946824, -0.0716107 , -0.09655498, -0.09650955,
       -0.01812868, -0.01823901, -0.03055686, -0.03071079,  0.05297628,
        0.05302816,  0.11288207,  0.11295395, -0.00912314, -0.00902285,
       -0.05166335, -0.05122344,  0.04606769,  0.04512527, -0.00179464,
       -0.00164097, -0.06021762, -0.06020006, -0.07943237, -0.07983376,
        0.02468139,  0.02467356, -0.3220949 ,  0.39084074,  0.28734684,
       -0.27217898,  0.01662595,  0.01668992, -0.04304055, -0.0430259 ,
       -0.08369417, -0.08379084,  0.02347496,  0.02346146, -0.02499558,
       -0.02499707,  0.07603627,  0.0759748 , -0.01914041, -0.01838751,
        0.06124983,  0.06123266, -0.07109787, -0.07115573,  0.05657353,
        0.05651395,  0.03263828,  0.03293543, -0.07934555, -0.07772869,
       -0.28702742,  0.38380033,  0.06065146,  0.06071574,  0.00731726,
        0.00732449, -0.04728706, -0.04729409, -0.06988908, -0.06846598,
        0.04680758,  0.04619215,  0.09211577,  0.09216671, -0.07270771,
       -0.07300868, -0.00184339, -0.00201967,  0.37051126, -0.35409802,
        0.01719776,  0.01385233,  0.03916124,  0.039269  ,  0.25239196,
       -0.21358725,  0.17249767, -0.22542334,  0.10825034,  0.1079891 ,
        0.00608608,  0.02335839,  0.04024133,  0.0402542 , -0.03098212,
       -0.03124478,  0.09769181,  0.09933688,  0.04818006,  0.04802442,
       -0.01126344, -0.01120145
   };

FCLayer::FCLayer (){
	//bias = new double[1];
	//input_size = 0;
	//output_size = 0;
}

FCLayer::FCLayer (int input_sz, int output_sz, 
					double scale, int zero_point,
					int batch, bool default_weight) {
	input_size = input_sz;
	output_size = output_sz;
	batch_size = batch;
	quant_zero_point = zero_point;
	quant_scale = scale;

	cout << "set input size to: " << input_size << "\n";
	cout << "set output size to: " << output_size << "\n";

	bias = new double[output_size];
	for(int j = 0; j < output_size; j++) {
		if(default_weight){
			bias[j] = initial_model_bias[j];
		}
		else{
			bias[j] = (double)rand()/(RAND_MAX);
		}
	}

	weights = new double*[input_size];
	for(int i = 0; i < input_size; i++) {
		weights[i] = new double[output_size];
		for(int j = 0; j < output_size; j++){
	    	if(default_weight){
	    		//initial_model_weights was 256x2 but now flattened
	    		//input_size = 256, output_size = 2
	    		weights[i][j] = initial_model_weights[j+i*output_size];
	    	}else{
	    		if(rand()%2 == 0){
	    			weights[i][j] = -1*(double)rand()/(RAND_MAX);
	    		}else{
	    			weights[i][j] = (double)rand()/(RAND_MAX);
	    		}
	    	}
			
	    }
	}

	output_error_softmax = new double*[batch_size];
	for(int b = 0; b < batch_size; b++) {
	    output_error_softmax[b] = new double[output_size];
	}
}


void FCLayer::set_weights_bias(double **new_weights, double *new_bias){
	// Set weights
	for(int i = 0; i < input_size; ++i) {
		for (int j = 0; j < output_size; ++j){
			weights[i][j] = new_weights[i][j];
		}		
	}

	// Set bias
	for(int j = 0; j < output_size; j++) {
		bias[j] = new_bias[j];
	}
}

void softmax(double *input_output_data, int classes){
	double sum = 0;
	double max = -10000.0;
	for(int j = 0; j < classes; j++){
		if(input_output_data[j] > max){
			max = input_output_data[j];
		}
	}
	for(int j = 0; j < classes; j++){
		input_output_data[j] = exp(input_output_data[j]-max);
		sum += input_output_data[j];
	}
	for(int j = 0; j < classes; j++){
		input_output_data[j] = input_output_data[j]/sum;
	}
}

void softmax_prime(double *pred, double *result, int classes){
	for(int a = 0; a < classes; a++){
		result[a] = 0;
		for(int b = 0; b < classes; b++){
			if(a == b){
				result[a] += pred[a]*(1-pred[a]);
			}else{
				result[a] += -1*pred[a]*pred[b];
			}
		}
	}
}

void FCLayer::dequantize(int *input_data, double *input_float){
	//use double input [input_size]; when calling
	//input = (input - zero_point) * scale
	for(int i = 0; i < input_size; i++){
		input_float[i] = (input_data[i]-quant_zero_point)*quant_scale;
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
				output[b][j] += input_float[b][i]*weights[i][j];
			}
			output[b][j] += bias[j];
		}
		//softmax to get probabilities
		softmax(output[b], output_size);
	}
}

double L2(double **weight, int input_size, int output_size, int batch_size){
	double sum = 0.0;
	for(int i = 0; i < input_size; i++){
		for(int j = 0; j < output_size; j++){
			sum += pow(weight[i][j],2);
		}
	}
	return sum / batch_size;
}

void cross_entropy_prime(double *pred, int *real, double *result, int classes){
	for(int j = 0; j < classes; j++){
		result[j] = -1.0*real[j]/pred[j];
	}
}

double cross_entropy_loss(double **pred, int **real, int batch_size, int classes){
	double result = 0.0;
	double epsilon = 0.0000001;

	for(int b = 0; b < batch_size; b++){
		for(int j = 0; j < classes; j++){
			if(real[b][j] == 1 and pred[b][j] < epsilon){
				result -= real[b][j]*log(epsilon);
			}else{
				result -= real[b][j]*log(pred[b][j]);
			}
		}
	}
	return result/(batch_size);
}

void mse_prime(double *pred, int *real, double *result, int classes){
	for(int i = 0; i < classes; i++){
		result[i] += 2*(pred[i]-real[i]);
	}

}

double mse(double **pred, int **real, int batch_size, int classes){
	double result = 0.0;

	for(int b = 0; b < batch_size; b++){
		for(int j = 0; j < classes; j++){
			result += pow((pred[b][j]-real[b][j]), 2);
		}
	}
	return result/(classes*batch_size);
}

void combined_ce_softmax_prime(double *pred, int *real, double *result, int classes){
	for(int i = 0; i < classes; i++){
		result[i] = pred[i] - real[i];
		//cout << result[i] << "\n";
		assert(result[i] >= -1 && result[i] <= 1);
	}
}

double accuracy(double **pred, int **real, int batch_size, int classes){
	double correct = 0.0;

	for(int b = 0; b < batch_size; b++){
		for(int j = 0; j < classes; j++){
			if((pred[b][j] >= 0.5 && real[b][j] == 1)||(pred[b][j] < 0.5 && real[b][j] == 0)){
				correct += 1;
			}
		}
	}
	return correct/(classes*batch_size);
}

void FCLayer::backward (double **output, int **ground_truth, 
						double **input_error, double **input_float,
						double learning_rate, double lambda) {

	//double **output_error = new double*[batch_size];

	for(int b = 0; b < batch_size; b++) {
	    //output_error[b] = new double[output_size];
	    //cross_entropy_prime(output[b], ground_truth[b], output_error[b], output_size);
	    //softmax_prime(output_error[b], output_error_softmax[b], output_size);
	    combined_ce_softmax_prime(output[b], ground_truth[b], output_error_softmax[b], output_size);
	}

	for(int b = 0; b < batch_size; b++){
		//input_error = np.dot(output_error, self.weights.T)
		// batch_sizexinput_size = batch_sizexoutput_size dot output_sizexinput_size
	    for(int i = 0; i < input_size; i++) {
	    	input_error[b][i] = 0.0;
		    for(int j = 0; j < output_size; j++){
				input_error[b][i] += output_error_softmax[b][j]*weights[i][j]/batch_size;
		    }
		}
	}

	//weights_error = np.dot(self.input.T, output_error)
	double weights_error[input_size][output_size];

	for(int i = 0; i < input_size; i++) {
		for(int j = 0; j < output_size; j++){
			weights_error[i][j] = 0.0;
		}
	}

	for(int i = 0; i < input_size; i++) {
		// input_sizexoutput_size = input_sizexbatch_size dot batch_sizexoutput_size
		for(int j = 0; j < output_size; j++){
			for(int b = 0; b < batch_size; b++){
				weights_error[i][j] += input_float[b][i]*output_error_softmax[b][j]/batch_size;
		    }
		}
	}

	//self.weights -= learning_rate * weights_error
	for(int i = 0; i < input_size; ++i) {
		for(int j = 0; j < output_size; j++){
			this->weights[i][j] = this->weights[i][j]*(1-learning_rate*lambda/batch_size)-learning_rate*weights_error[i][j];
		}
	}
	
	//self.bias -= learning_rate * output_error
	for(int j = 0; j < output_size; j++){
		for(int b = 0; b < batch_size; b++) {
			this->bias[j] -= learning_rate*output_error_softmax[b][j]/batch_size;
		}
	}
}

void FCLayer::cleanup(){
	cout << "cleaning up object\n";
	for(int i = 0; i < input_size; i++) {
		delete [] weights[i];
	}
	for(int b = 0; b < batch_size; b++){
	    delete [] output_error_softmax[b];
	}
	delete [] output_error_softmax;
	delete[] weights;
	delete[] bias;
}

FCLayer::~FCLayer() { }

void predict(double **input_float, FCLayer *model, double **output, int batch_size, int classes){
	model->forward(input_float, output);

	for(int b = 0; b < batch_size; b++){
		for(int j = 0; j < classes; j++){
			if(output[b][j] >= 0.5){
				output[b][j] = 1;
			}else{
				output[b][j] = 0;
			}
		}
	}
}

void FL_round_simulation(double **input_f, int **input_i, int **ground_truth, int local_episodes, 
						double learning_rate, FCLayer *model, double lambda, 
						bool verbose, bool local, bool unquantize){
	if(verbose == true){
		if(local == true){
			cout << "\tstarted sim\n";
		}else{
			//Serial.print("\tstarted sim\n");
		}
	}
	double **output = new double*[model->batch_size];
	double **input_error = new double*[model->batch_size];
	double **input_float = new double*[model->batch_size];

	if(verbose == true){
		if(local == true){
			cout << "\tallocated\n";
		}else{
			//Serial.print("\tallocated\n");
		}
	}

	for(int b = 0; b < model->batch_size; b++) {
		output[b] = new double[model->output_size];
		for(int j = 0; j < model->output_size; j++){
			output[b][j] = 0.0;
		}

		input_error[b] = new double[model->input_size];
		for(int i = 0; i < model->input_size; i++){
			input_error[b][i] = 0.0;
		}
		if(unquantize == true){
			model->dequantize(input_i[b], input_float[b]);
		}else{
			input_float[b] = input_f[b];
		}
	}

	if(verbose == true){
		if(local == true){
			cout << "\tallocated part 2\n";
			cout << "\tinitial bias loaded " << model->bias[0] << " " << model->bias[1] << "\n";
			cout << "\tinitial output " << output[0][0] << " " << output[0][1] << "\n";
		}else{
			//Serial.print("\tallocated\n");
		}
	}

	for(int epi = 0; epi < local_episodes; epi++){
		if(verbose == true){
			if(local == true){
				cout << "EPISODE " << epi << "\n";
				cout << "\tforward\n";
				cout << "\t\tbias loaded " << model->bias[0] << " " << model->bias[1] << "\n";
			}else{
				//Serial.print("\tEPISODE");
        		//Serial.println(epi);
        		//Serial.print("\tforward\n\t\t bias loaded");
				//Serial.print(model->bias[0], model->bias[1]);
			}
		}
		//forward
		model->forward(input_float, output);

		if(verbose == true){
			if(local == true){
				cout << "\tsoftmax " << output[0][0] << " " << output[0][1] << "\n";
			}else{
				//Serial.print("\tsoftmax");
				//Serial.println(output[0][0]);
				//Serial.println(output[0][1]);
			}
		}

		//calculate and print error
		double l2_error = lambda*L2(model->weights, model->input_size, model->output_size, model->batch_size)/2;
		double error = cross_entropy_loss(output, ground_truth, model->batch_size, model->output_size)
						+l2_error;
		double acc = accuracy(output, ground_truth, model->batch_size, model->output_size);
		cout << "\tlocal episode : " << epi << " error: " << error << " accuracy: " << acc << "\n";

		//backward
		model->backward(output, ground_truth, input_error, input_float, learning_rate, lambda);

		if(verbose == true){
			if(local == true){
				cout << "\tbackward\n";
			}else{
				//Serial.print("\tbackward\n");
			}
		}
		//reset input_error and output in forward and backward 
	}

	if(verbose == true){
		if(local == true){
			cout << "\tdone loop\n";
		}else{
			//Serial.print("\tdone loop\n");
		}
	}

	for(int b = 0; b < model->batch_size; b++) {
		delete [] input_error[b];
		delete [] output[b];
		if(unquantize == true){
			delete [] input_float[b];
		}
	}
	delete [] input_error;
	delete [] output;
	delete [] input_float;

	if(verbose == true){
		if(local == true){
			cout << "\tdone de-allocation\n";
		}else{
			//Serial.print("\tdone de-allocation\n");
		}
	}
}