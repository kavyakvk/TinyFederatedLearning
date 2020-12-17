/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
const bool real_world = false; // change this if want to use Arducam
static int current_round = 0;
static char readEmbeddingsString[input_size * batch_size * (sizeof(double) / sizeof(char))];
static char readTruthsString[batch_size * output_size * (sizeof(double) / sizeof(char))];
//static char readWeightsString[input_size * output_size * (sizeof(double) / sizeof(char))];
static char readWeightsString[6000];

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
}  // namespace



// For getting embeddings and weights
static int ndx = 0;  // For keeping track where in the char array to input
static int numChars = input_size * batch_size;
static bool endOfResponse = false; 
static char * pch;
static char * pch_row;


char readc_blocking() {
   while (Serial.available() < 1)
      ; // spin/block, waiting for data
   return Serial.read();
}


// The name of this function is important for Arduino compatibility.
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

// The name of this function is important for Arduino compatibility.
void loop() {
  Serial.println("Connected");
  delay(5000);
  
  if(real_world){
    // Get image from provider.
    if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                              input->data.int8)) {
      TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
    }

    Serial.println("captured image finished.");
    Serial.println("inference starting.");

    // Run the model on this input and make sure it succeeds.
    if (kTfLiteOk != interpreter->Invoke()) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
    }

    Serial.println("Inference finished.");

    // Get the embedding from the feature extractor
    TfLiteTensor* output = interpreter->output(0);

    //  TF_LITE_REPORT_ERROR(error_reporter,"Char array: %d", output->data.uint8);

    RespondToDetection(error_reporter, 10, -10);

    // Process the inference results.
    // Embedding
    //int8_t person_score = output->data.uint8[kPersonIndex];
    //int8_t no_person_score = output->data.uint8[kNotAPersonIndex];
    for(int i = 0; i < 256; i++){
      Serial.print(output->data.uint8[i]);
      Serial.print(" ");
    }
    Serial.println();
    //  RespondToDetection(error_reporter, person_score, no_person_score);


  } else{
    current_round += 1;
    
    //FOR EVERY DEVICE 
    int epochs = 3;

    float weights[input_size][output_size];
    float bias[output_size];

    Serial.print("creating devices \n");

    //FCLayer hi = FCLayer(input_size, output_size, quant_scale, quant_zero_point, batch_size, true);

    FCLayer devices[fl_devices];

    for(int i = 0; i < fl_devices; i++){
      devices[i] = FCLayer(input_size, output_size, quant_scale, quant_zero_point, batch_size, true);
    
      FCLayer* NNmodel = &(devices[i]);

      // // GET EMBEDDINGS
      // //Get embedding, save into float** input_data (batch_size x input_size) and int** ground_truth (batch_size x ouput_size)
      // readEmbeddingsString[0] = '\0';   // reset
      // ndx = 0;
      // endOfResponse = false;
      // // Reading in embedding
      // while(!Serial.available()) {delay(500);} // wait for data to arrive
      // // serial read section
      // while (Serial.available() > 0 || endOfResponse == false)
      // {
      //   if (Serial.available() > 0)
      //   {
      //     char c = Serial.read();  //gets one byte from serial buffer
      //     readEmbeddingsString[ndx] = c; //adds to the string
      //     ndx += 1;
      //     if (c == '\n'){
      //       endOfResponse = true;
      //       Serial.println("A: Finished reading in embedding");
      //     }
      //   }
      // }

      // delay(500);

      // // Tokenize string and split by commas
      // Serial.println("Splitting string for embedding");
      // double embeddings_arr[numChars];
      // pch = strtok (readEmbeddingsString, ",;");
      // int embedding_index = 0;
      // while (pch != NULL){
      //   embeddings_arr[embedding_index] = atof(pch);  // Store in array
      //   pch = strtok (NULL, ",;");   // NULL tells it to continue reading from where it left off
      //   embedding_index += 1;
      // }
      // Alternatively
      // for(int b = 0; i < NNmodel->batch_size; b++){
      //   for(int j = 0; j < NNmodel->input_size; j++){
      //     if(i == 0 & j == 0){
      //       pch = strtok (readWeightsString, ",;");
      //     }
      //     else{
      //       pch = strtok (NULL, ",;");
      //     }
      //     input_data[b][j] = atof(pch);
      //   }
      // }

      //save embeddings and ground truths to an array
      Serial.print("Splitting string for embedding\n");

      double embeddings_arr[batch_size*input_size];
      int embedding_index = 0;
      for(int i = 0; i < batch_size*input_size; i++){
        embeddings_arr[embedding_index] = (double)rand()/(RAND_MAX);
        embedding_index += 1;
      }

      Serial.print("finished Splitting string for embedding\n");

      // // GET GROUND TRUTHS
      // //Get int** ground_truth (batch_size x ouput_size)
      // readTruthsString[0] = '\0';   // reset
      // ndx = 0;
      // endOfResponse = false;
      // // Reading in ground truths
      // while(!Serial.available()) {delay(500);} // wait for data to arrive
      // // serial read section
      // while (Serial.available() > 0)
      // {
      //   if (Serial.available() > 0)
      //   {
      //     char c = Serial.read();  //gets one byte from serial buffer
      //     readTruthsString[ndx] = c; //adds to the string
      //     ndx += 1;
      //     if (c == '\n'){
      //       endOfResponse = true;
      //       Serial.println("A: Finished reading in ground truths");
      //     }
      //   }
      // }

      // delay(500);

      // // Tokenize string and split by commas
      // Serial.println("Splitting string for ground truths");
      // int gt_arr[batch_size * output_size];
      // pch = strtok (readTruthsString, ",;");
      // int gt_index = 0;
      // while (pch != NULL){
      //   gt_arr[gt_index] = atoi(pch);  // Store in array
      //   pch = strtok (NULL, ",;");   // NULL tells it to continue reading from where it left off
      //   gt_index += 1;
      // }
      // Alternatively
      // for(int b = 0; i < NNmodel->batch_size; b++){
      //   for(int j = 0; j < NNmodel->output_size; j++){
      //     if(i == 0 & j == 0){
      //       pch = strtok (readTruthsString, ",;");
      //     }
      //     else{
      //       pch = strtok (NULL, ",;");
      //     }
      //     ground_truth[b][j] = atoi(pch);
      //   }
      // }

      Serial.print("Splitting string for gt\n");
      int gt_arr[batch_size * output_size];
      for(int i = 0; i < output_size*batch_size; i++){
          gt_arr[i] = (i % 2);
      }
      
      Serial.print("finished Splitting string for gt\n");

      double **input_data = new double*[NNmodel->batch_size];
      int **ground_truth = new int*[NNmodel->batch_size];
      Serial.print("finished allocating input_data and ground_truth \n");
      for(int b = 0; b < NNmodel->batch_size; b++) {
        input_data[b] = new double[NNmodel->input_size];
        ground_truth[b] = new int[NNmodel->output_size];
        Serial.print("finished allocating actual arrays for input_data and ground_truth \n");
        for(int i = 0; i < NNmodel->input_size; i++){
          input_data[b][i] = embeddings_arr[b*input_size + i];
        }
        for (int j = 0; j < NNmodel->output_size; j++){
          ground_truth[b][j] = gt_arr[b*output_size + j];
        }
      }

      Serial.println("finished loading data into input data and ground truth");
      // GET WEIGHTS double** (input_size x output_size x)
//      memset(readWeightsString, 0, sizeof(readWeightsString));     // reset
      readWeightsString[0] = '\0';
      int ndx_test = 0;
      endOfResponse = false;
//      while(!Serial.available()) {} // wait for datandx to arrive
      // serial read section
//      while (ndx_test < 300)
//      {
////        if (Serial.available() > 0)
////        {
//        char c = readc_blocking();
////        Serial.print
////        char c = Serial.read();  //gets one byte from serial buffer
//        readWeightsString[ndx_test] = c; //adds to the string
//        ndx_test++;
//        if (c == '\n'){
//          endOfResponse = true;
//          Serial.println("A: Finished reading in weights and bias");
//        }
////        }
//      }
      // Tokenize string and split by commas
      // Serial.println(ndx);
//      Serial.println(readWeightsString);
      while(!Serial.available()){}    // Wait for serial input
      String test_read = Serial.readString();
      Serial.println(test_read);
      Serial.println(test_read.length());
      Serial.println("Splitting string for weights");
//      Serial.println(strtok(readWeightsString, ",;\n"));
      double **init_weights = new double*[NNmodel->input_size];
      for(int i = 0; i < NNmodel->input_size; i++){
        init_weights[i] = new double[NNmodel->output_size];
        for(int j = 0; j < NNmodel->output_size; j++){
          if(i == 0 && j == 0){
            pch = strtok (readWeightsString, ",;\n");
          }
          else{
            pch = strtok (NULL, ",;\n");
          }
          init_weights[i][j] = (double)atof(pch);
          Serial.print(init_weights[i][j]);
          Serial.print(" ");
        }
      }
      // Tokenize string for bias
      Serial.println("Splitting string for bias");
      double *init_bias = new double[NNmodel->output_size];
      for(int i = 0; i < NNmodel->output_size; i++){
        pch = strtok (NULL, ",;");  // leftover from weights string
        init_bias[i] = atof(pch);
        Serial.print(init_bias[i]);
        Serial.print(" ");
      }

      Serial.println("Finished loading in and setting weights + bias");

//      NNmodel->set_weights_bias(init_weights, init_bias);
//
//      FL_round_simulation(input_data, ground_truth, local_epochs, 0.01, NNmodel, true, false);
//
//      Serial.println("Finished simulation");

      NNmodel->cleanup();

      for(int b = 0; b < batch_size; b++) {
        delete [] input_data[b];
        delete [] ground_truth[b];
      }
      delete [] input_data;
      delete [] ground_truth;
      for (int i = 0; i < NNmodel->input_size; i++) {
        delete [] init_weights[i];
      }
      delete [] init_weights;
      delete [] init_bias;
    }

    

    //Average weights
    // for(int d = 0; d < fl_devices; d++){
    //   FCLayer* NNmodel = &(devices[d]);
    //   for(int j = 0; j < output_size; j++){
    //     for(int i = 0; i < input_size; i++){
    //       weights[i][j] += NNmodel->weights[i][j]
    //     }
    //   }
    // }

    // for(int j = 0; j < output_size; j++){
    //   for(int i = 0; i < input_size; i++){
    //     weights[i][j] = weights[i][j]/fl_devices;
    //   }
    //   bias[j] = bias[j]/fl_devices;
    // }

    //send weights to server
    
  }
}
