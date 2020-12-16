/* TinyFedTL
 *  Eric Lin and Kavya Kopparapu
==============================================================================*/

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "FCLayer.h"
#include <vector>
//#include "FL_model_weights.h"

//#include "NeuralNetwork.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
static std::vector<FCLayer> devices;

// What we added
//CONSTANTS FOR FEDERATED LEARNING
const int input_size = 256;
const int output_size = 2;
const double quant_scale = 0.04379776492714882;
const int quant_zero_point = -128;
const int fl_devices = 1;
const int batch_size = 1;
const int local_epochs = 20;
static int current_round = 0;
// static char readEmbeddingsString[input_size * batch_size * (sizeof(double) / sizeof(char))];
// static char readTruthsString[batch_size * output_size * (sizeof(double) / sizeof(char))];
//static char readWeightsString[input_size * output_size * (sizeof(double) / sizeof(char))];
static char readWeightsString[5900];

// In order to use optimized tensorflow lite kernels, a signed int8 quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
// constexpr int kTensorArenaSize = 115656;
constexpr int kTensorArenaSize = 135000;
static uint8_t tensor_arena[kTensorArenaSize];
}  // end namespace


// For getting embeddings and weights
static int ndx = 0;  // For keeping track where in the char array to input
// static int numChars = input_size * batch_size;
bool endOfResponse = false; 
char * pch;
// static char * pch_row;

void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_AVERAGE_POOL_2D,
                               tflite::ops::micro::Register_AVERAGE_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

  //INITIALIZE THE MODEL
  
  for(int i = 0; i < fl_devices; i++)
  {
    devices.push_back(FCLayer(input_size, output_size, quant_scale, quant_zero_point, batch_size, true));
  }
  
  Serial.begin(9600);  // initialize serial communications at 9600 bps
}

void loop() {  
  // From Arduino
  // Get image from provider.
  if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                            input->data.int8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
  }
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }
  TfLiteTensor* output = interpreter->output(0);
  // Process the inference results.
  int8_t person_score = output->data.uint8[kPersonIndex];
  int8_t no_person_score = output->data.uint8[kNotAPersonIndex];
  RespondToDetection(error_reporter, person_score, no_person_score);
  // End Arduino

  // Receive weights
  if(endOfResponse == false){
    Serial.println("Start");
  } 
  ndx = 0; 
  endOfResponse = false;
  while(endOfResponse == false){
    while(!Serial.available()) {} // wait for data to arrive
    // serial read section
    while (Serial.available() > 0)
    {
      char c = Serial.read();  //gets one byte from serial buffer
      readWeightsString[ndx] = c; //adds to the string
      ndx += 1;
      if (c == '|'){
        endOfResponse = true;
      }
    }
    Serial.print(ndx);
  }

  // Devices
  FCLayer devices[fl_devices];

  for(int i = 0; i < fl_devices; i++){
    devices[i] = FCLayer(input_size, output_size, quant_scale, quant_zero_point, batch_size, true);
  }
  FCLayer* NNmodel = &(devices[0]);

  double **init_weights = new double*[NNmodel->input_size];
  double *init_bias = new double[NNmodel->output_size];
  // Train then send new weights back
  if (endOfResponse == true)
  {
    // Split into weights_arr
    pch = strtok (readWeightsString, ",");
    // int embedding_index = 0;
    // while (pch != NULL){
    //   weights_arr[embedding_index] = (double) atof(pch);  // Store in array
    //   pch = strtok (NULL, ",");   // NULL tells it to continue reading from where it left off
    //   embedding_index += 1;
    // }
    // Tokenize string for weights
    for(int i = 0; i < NNmodel->input_size; i++){
      init_weights[i] = new double[NNmodel->output_size];
      for(int j = 0; j < NNmodel->output_size; j++){
        init_weights[i][j] = (double)atof(pch);
        pch = strtok (NULL, ",");
      }
    }
    // Tokenize string for bias
    for(int i = 0; i < NNmodel->output_size; i++){
      init_bias[i] = (double) atof(pch);
      pch = strtok (NULL, ",;");  // leftover from weights string
    }

    // Set weights
    NNmodel->set_weights_bias(init_weights, init_bias);

    // Train
    FL_round_simulation(input_data, ground_truth, local_epochs, 0.01, NNmodel, true, false);
    Serial.println("Finished simulation");


    // Print out new weights and bias
    for(int i = 0; i < NNmodel->input_size; i++){
      for(int j = 0; j < NNmodel->output_size; j++){
        Serial.print(init_weights[i][j], 8);
        Serial.print(" ");
      }
    }
    for(int i = 0; i < NNmodel->output_size; i++){
      Serial.print(init_bias[i], 8);
      Serial.print(" ");
    }
    
  }

  NNmodel->cleanup();

  for (int i = 0; i < NNmodel->input_size; i++) {
    delete [] init_weights[i];
  }
  delete [] init_weights;
  delete [] init_bias;

  delay(5000);
  Serial.flush();

}
