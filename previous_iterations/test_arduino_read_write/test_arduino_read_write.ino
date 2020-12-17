// Serial test script

static int ndx = 0;  // For keeping track where in the char array to input
//const int numChars = 3000;
static char readString[6000];
bool endOfResponse = false; 
static double weights_arr[515] = { };     // initialize all elements to 0
char * pch;

//void ack() {
//  readString = "";
//  while (waitForResponse) {
//    while(!Serial.available()) {} // wait for data to arrive
//    // serial read section
//    while (Serial.available())
//    {
//      
//    }
//  }
//}

void setup()
{

  Serial.begin(9600);  // initialize serial communications at 9600 bps
  Serial.println("please print");
}

void loop()
{
  if(endOfResponse == false){
    Serial.println("Start");
  } 
  ndx = 0; 
  endOfResponse = false;
//  readString[0] = '\0';   // reset
  while(endOfResponse == false){
//    for(int i = 0; i < 4; i++){
//    Serial.print("Ndx ");
    while(!Serial.available()) {} // wait for data to arrive
    // serial read section
    while (Serial.available() > 0)
    {
      char c = Serial.read();  //gets one byte from serial buffer
      readString[ndx] = c; //adds to the string
      ndx += 1;
      if (c == '|'){
        endOfResponse = true;
//        Serial.println("Reached end of string");
      }
    }
    Serial.print(ndx);
//    Serial.print("Read in ");
//    Serial.print(ndx);
//    Serial.print(" chars: ");
//    Serial.println(readString);
  }
//  Serial.print("Read in ");
//  Serial.print(ndx);
//  Serial.print(" chars: ");
//  Serial.println(readString);
  
//  Serial.println("read serial");


  if (endOfResponse == true)
  {
//    Serial.print("Arduino: received ");  
//    Serial.println(readString); //see what was received
//      float float_received = readString.toFloat();
//      if(float_received == (float) 0.63){
//        Serial.print("Arduino: received ");
//        Serial.println(float_received);
//      }
//    if(endOfResponse){
//      Serial.println("A: Reached end of response");
//    }
    // Tokenize string and split by commas
//    Serial.println("Splitting string for embedding");
    pch = strtok (readString, ",");
    int embedding_index = 0;
    while (pch != NULL){
//      Serial.println(pch);
      weights_arr[embedding_index] = (double) atof(pch);  // Store in array
      pch = strtok (NULL, ",");   // NULL tells it to continue reading from where it left off
      embedding_index += 1;
//      if(embedding_index % 50 == 0){
//        Serial.println(embedding_index);
//      }
    }

    // Print out array
//    Serial.print("Embeddings: [");
    for (int i = 0; i < (int)(sizeof(weights_arr)/sizeof(weights_arr[0])); i++){
      Serial.print(weights_arr[i], 7);    // 6 decimal points
      Serial.print(" ");
    }
//    Serial.println("]");
    
  }

  delay(5000);

  // serial write section
//  char ard_sends = '1';
//  Serial.print("Arduino: sent ");
//  Serial.println(ard_sends);
//  Serial.print("\n");
  Serial.flush();
}
