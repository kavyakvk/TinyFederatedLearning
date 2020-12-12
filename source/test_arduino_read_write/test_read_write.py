import serial
import syslog
import time


def ack_handshake(ard):
    '''
        Establishes a 3-step handshake with the Arduino
    '''
    ack = False
    time.sleep(5)
    ard.flush()

    # Send ack
    ard.write('ACK'.encode())
    print("Mac: sent ACK")
    for i in range(5):  # try 5 times to receive ACK
        time.sleep(1)
        ard.flush()
        msg = ard.read(ard.inWaiting())
        print(f"Mac: received {msg}")
        if msg.decode('utf-8') == 'ACK':
            ack = True
            # Send last ack
            print("Mac: sent ACK-2")
            ard.write('ACK-2'.encode())
            break

    print(f'Ack handshake success? {ack}')
    return ack



def main():
    #The following line is for serial over GPIO
    port = '/dev/cu.usbmodem142301' # change this to what the Arduino Port is
    ard = serial.Serial(port,9600,timeout=5)

    test_embeddings = [
        "0.63,0.68,0.12,0.12,0.12,-0.1\n",
        "0.1,0.0,0,0.8\n",
        "3",
        "4",
        "5"
    ]

    time.sleep(5) # wait for Arduino


    i = 0

    while (i < 2):
        # Serial write section
        ard.flush()
        print("Mac: sent embeddings")
        ard.write(test_embeddings[i].encode())
        time.sleep(1) 

        # Serial read section
        msg = ard.read(ard.inWaiting()) # read all characters in buffer
        print(f"Mac: received {msg}")
        # print (f"Mac: received {msg.decode('utf-8')}")
        i = i + 1
    else:
        print ("Exiting")
    exit()



# Embeddings (every image) pickle into a file 
# Model weights (after 5 in a row) averaged together and then send back to arduino

# 1. if we send data (embeddings) to arduino
# 2. if we get embeddings from arduino, that means we need to wait like 30 seconds for each pic


if __name__ == "__main__":
    main()