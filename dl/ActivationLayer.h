#ifndef NN_ACT_H
#define NN_ACT_H

class ActivationLayer {
    int input_size, output_size, batch_size, quant_zero_point;
    double quant_scale;

  public:
    ActivationLayer ();
    double forward (double **input_float, double **output);
    double backward (double output_error, double learning_rate); 
    ~ActivationLayer();
};