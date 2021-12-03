# TinyFederatedLearning
TinyML has rose to popularity in an era where data is everywhere. However, the data that is in most demand is subject to strict privacy and security guarantees. In addition, the deployment of TinyML hardware in the real world has significant memory and communication constraints that traditional ML fails to address. In light of these challenges, we persent TinyFedTL, the first implementation of federated transfer learning on a resource- constrained microcontroller. 

* C++ implementation to accompany Arduino code is provided (without standard library so it will run on-device!)
* Please see our demo video presentation here: https://www.youtube.com/watch?v=KSaidr3ZN9M

## File Structure and Important Files
* dl
* source > arduino_training_final_v3: the .ino file has the implementation of our FL code for the Arduino IDE to compile
  * python_final_script.py acts as the "central server" for the arduino
* source > simulation: 
  * NeuralNetwork.cpp has our FC implementation and the FL implementation, simulation.cc is the file with the code necessary to run our simulations.
  * simulation-xxx are the executables that can be run with ./ for each of our experiments, and the .txt files are the output from terminal when running the experiments
  * fl_simulation_analysis generates the .csv from the .txt
  * graphing.ipynb has the information to graph our figures from the paper from the .csv files
* tensorflow (no changes)
* third_party (no changes)
