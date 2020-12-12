#ifndef NN_FC_H
#define NN_FC_H

class FCLayer {
    public: 
        int input_size, output_size, batch_size, quant_zero_point;
        double quant_scale;
        double *bias;
        double **weights;

        FCLayer();
        FCLayer (int input_sz, int output_sz, double scale, 
        			int zero_point, int batch, bool default_weight);
        void forward (double **input_float, double **output);
        void backward (double **output, int **ground_truth, 
    						double **input_error, double **input_float,
    						double learning_rate); 
        void dequantize(int *input_data, double *input_float);
        void cleanup();
        ~FCLayer();
        //void set_weights(double **new_weights); //NEEDS TO BE FIXED
};

void FL_round_simulation(double **input_float, int **ground_truth, int local_episodes, 
                        double learning_rate, FCLayer *model, bool verbose, bool local);

void FL_round_quantize(int **input_data, int **ground_truth, int local_epochs, 
                double learning_rate, FCLayer *model);


#endif
