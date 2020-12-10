// Serial test script

static byte ndx = 0;  // For keeping track where in the char array to input
const byte numChars = 256;
char readString[numChars];
bool endOfResponse = false; 
float embeddings_arr[10] = { };     // initialize all elements to 0
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

}

void loop()
{
  readString[0] = '\0';   // reset
  ndx = 0;
  while(!Serial.available()) {} // wait for data to arrive
  // serial read section
  while (Serial.available() > 0 && endOfResponse == false)
  {
    if (Serial.available() >0)
    {
      char c = Serial.read();  //gets one byte from serial buffer
      readString[ndx] = c; //adds to the string
      ndx += 1;
      if (c == '\n'){
        endOfResponse = true;
      }
    }
  }


  if (strlen(readString) > 0)
  {
    Serial.print("Arduino: received ");  
    Serial.println(readString); //see what was received
//      float float_received = readString.toFloat();
//      if(float_received == (float) 0.63){
//        Serial.print("Arduino: received ");
//        Serial.println(float_received);
//      }
    if(endOfResponse){
      Serial.println("A: Reached end of response");
    }
    // Tokenize string and split by commas
    Serial.println("Splitting string for embedding");
    pch = strtok (readString, ",");
    int embedding_index = 0;
    while (pch != NULL){
      Serial.println(pch);
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
    
  }

  delay(500);

  // serial write section
  char ard_sends = '1';
  Serial.print("Arduino: sent ");
  Serial.println(ard_sends);
  Serial.print("\n");
  Serial.flush();
}
