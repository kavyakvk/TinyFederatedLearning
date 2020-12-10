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

//#include "NeuralNetwork.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

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

//CONSTANTS FOR FEDERATED LEARNING
const int input_size = 256;
const int output_size = 2;
const double quant_scale = 0.04379776492714882;
const int quant_zero_point = -128;
const int fl_devices = 3;
const int batch_size = 5;
const int local_epochs = 3;

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
  static std::vector<FCLayer> devices;
  for(int i = 0; i < fl_devices; i++)
  {
    devices.push_back(FCLayer(input_size, output_size, quant_scale, quant_zero_point, batch_size, TRUE));
  }
  static bool real_world = FALSE; // change this if want to use Arducam
  static int current_round = 0;

  // For getting embeddings and weights
  static byte ndx = 0;  // For keeping track where in the char array to input
  const byte numChars = 256;
  char readString[numChars];
  bool endOfResponse = false; 
  float embeddings_arr[10] = { };     // initialize all elements to 0
  char * pch;
  Serial.begin(9600);  // initialize serial communications at 9600 bps
}

// The name of this function is important for Arduino compatibility.
void loop() {
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

    Serial.println("inference finished.");

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

    for(int d = 0; d < fl_devices; d++){
      FCLayer* NNmodel = &(devices[d]);
      
      //Get embedding, save into float** input_data (batch_size x input_size) and int** ground_truth (batch_size x ouput_size)
      readString[0] = '\0';   // reset
      ndx = 0;
      endOfResponse = false;
      // Reading in embedding
      while(!Serial.available()) {} // wait for data to arrive
      // serial read section
      while (Serial.available() > 0 && endOfResponse == false)
      {
        if (Serial.available() > 0)
        {
          char c = Serial.read();  //gets one byte from serial buffer
          readString[ndx] = c; //adds to the string
          ndx += 1;
          if (c == '\n'){
            endOfResponse = true;
            Serial.println("A: Finished reading in embedding");
          }
        }
      }

      delay(500);

      // Tokenize string and split by commas
      Serial.println("Splitting string for embedding");
      pch = strtok (readString, ",");
      int embedding_index = 0;
      while (pch != NULL){
        embeddings_arr[embedding_index] = atof(pch);  // Store in array
        pch = strtok (NULL, ",");   // NULL tells it to continue reading from where it left off
        embedding_index += 1;
      }

      // Print out array
      Serial.print("Embeddings: [");
      for (byte i = 0; i < (sizeof(embeddings_arr)/sizeof(embeddings_arr[0])); i++){
        Serial.print(embeddings_arr[i]);
        Serial.print(" ");
      }
      Serial.println("]");

      //save embeddings to an array
      double **input_data = new double*[NNmodel->batch_size];
      int **ground_truth = new int*[NNmodel->batch_size]
      for(int b = 0; b < NNmodel->batch_size; b++) {
        input_data[b] = new double*[NNmodel->input_size];
        ground_truth[b] = new double*[NNmodel->output_size];
        for(int i = 0; i < NNmodel->input_size; i++){
          input_data[b][i] = INPUT THE VALUE OF THE BTH EMBEDDING AT INDEX I;
          ground_truth[b][i] = INPUT THE VALUE OF THE BTH GROUND TRUTH AT INDEX I;
        }
      }


      //Get model weights from server
      // read weight into weights and the bias bit into bias
      NNmodel->set_weights(); //set weight of model here by passing double**
      
      //Train FL Round
      FL_round_simulation(input_data, ground_truth, local_epochs, 0.01, NNmodel);

      //update learning rate

      //de-allocate the memory stored
      for(int b = 0; b < NNmodel->batch_size; b++) {
        delete [] input_data[b];
        delete [] ground_truth[b];
      }
    }

    //Average weights
    for(int d = 0; d < fl_devices; d++){
      FCLayer* NNmodel = &(devices[d]);
      for(int j = 0; j < output_size; j++){
        for(int i = 0; i < input_size; i++){
          weights[i][j] += NNmodel->weights[i][j]
        }
      }
    }

    for(int j = 0; j < output_size; j++){
      for(int i = 0; i < input_size; i++){
        weights[i][j] = weights[i][j]/fl_devices;
      }
      bias[j] = bias[j]/fl_devices;
    }

    //send weights to server
    
  }
}
