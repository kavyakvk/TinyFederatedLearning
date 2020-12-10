#ifndef NN_FC_H
#define NN_FC_H

class FCLayer {
    public: 
        int input_size, output_size, batch_size, quant_zero_point;
        double quant_scale;
        double *bias;
        double **weights;

        FCLayer (int input_size, int output_size, double quant_scale, 
        			int zero_point, int batch, bool default_weight);
        void forward (double **input_float, double **output);
        void backward (double **output, double **ground_truth, 
    						double **input_error, double **input_float,
    						double learning_rate); 
        void dequantize(int *input_data, double *input_float);
        ~FCLayer();
        void set_weights(double **new_weights);
};

#endif

#ifndef NN_FC_INITIAL_MODEL_WEIGHTS
#define NN_FC_INITIAL_MODEL_WEIGHTS

extern const double initial_model_weights[];

#endif
