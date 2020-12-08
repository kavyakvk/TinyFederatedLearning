class FCLayer {
    int input, output;

  public:
    FCLayer (int,int);
    int forward (int input_data);
    float backward (float output_error, float learning_rate); 
};