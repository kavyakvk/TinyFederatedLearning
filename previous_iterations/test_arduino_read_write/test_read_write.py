import serial
import syslog
import time
import pickle


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

    # Initial Weights
    init_weights = pickle.load(open('../../dl/pickle_initial_model_weights.p', 'rb'))
    init_bias = [0.08060145, -0.08060154];          # bias (initial)
    init_weights.extend(init_bias)                  # extend bias to end of 1d weights array
    init_weights_str = (",".join(map(str, init_weights)) + "|").encode()
    print(init_weights_str)
    len_weights_str = len(init_weights_str)
    weights_iters = len_weights_str // 255
    print(len_weights_str)
    print(len(init_weights))

    # test_embeddings = [
    #     "0.63,0.68,0.12,0.12,0.12,-0.1\n",
    #     "0.1,0.0,0,0.8\n",
    #     "3",
    #     "4",
    #     "5"
    # ]

    time.sleep(3) # wait for Arduino


    i = 0

    while (i < 1):
        # Serial write section
        # ard.flush()
        # print("Mac: sent embeddings")
        # ard.write(test_embeddings[i].encode())
        # time.sleep(1) 
        
        for j in range(weights_iters):
            # Serial write section
            ard.flush()
            # print("Mac: sent weights and bias")
            ard.write(init_weights_str[j*255:(j+1)*255])

            time.sleep(1)

            # Read Arduino's response
            msg = ard.read(ard.inWaiting()) # read all characters in buffer
            print(f"Arduino received {msg.decode('utf-8')} out of {len_weights_str} bytes")

        # Send last remaining chars
        if weights_iters * 255 < len_weights_str:
            print(f"last remaining: {init_weights_str[weights_iters*255:]}")
            ard.write(init_weights_str[weights_iters*255:])

        output_weights_str = ''
        num_packets = 8
        for j in range(num_packets):
            time.sleep(5)
            # Read Arduino's response
            msg = ard.read(ard.inWaiting()) # read all characters in buffer
            print(f"Server: receiving weight packet {j} out of {num_packets}")
            output_weights_str += msg.decode("utf-8")
            # print(f"Mac: received {output_weights_str}")
        # print(output_weights_str)
        output_weight_lst = list(map(float, output_weights_str.split()))
        print(output_weight_lst)
        print(len(output_weight_lst))
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