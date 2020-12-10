class FCLayer {
    int input, output;

  public:
    FCLayer (int input_size, int output_size, double quant_scale, double zero_point);
    double forward (int input_data);
    double backward (double output_error, double learning_rate); 
    ~FCLayer();
};