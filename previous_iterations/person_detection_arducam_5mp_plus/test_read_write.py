import serial
import syslog
import time

#The following line is for serial over GPIO
port = '/dev/cu.usbmodem142101' # change this to what the Arduino Port is


ard = serial.Serial(port,9600,timeout=5) # default is baud 9600
time.sleep(5) # wait for Arduino to load and connect to port

i = 0

while (i < 4):
    # Serial write section

    setTempCar1 = 63
    setTempCar2 = 37
    ard.flush()
    setTemp1 = str(setTempCar1)
    setTemp2 = str(setTempCar2)
    print ("Python value sent: ")
    print (setTemp1)
    ard.write(setTemp1)
    time.sleep(1) # Match the Arduino code

    # Serial read section
    msg = ard.read(ard.inWaiting()) # read all characters in buffer
    print ("Message from arduino: ")
    print (msg)
    i = i + 1
else:
    print("Exiting")
exit()