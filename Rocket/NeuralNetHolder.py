import nn
from Try_Preprocessing import Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
class NeuralNetHolder:

    def __init__(self):
        super().__init__()
    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        delimiter = ","
        max_val1 = 812.1764
        min_val1 = -812.103
        max_val2 = 840.45557
        min_val2 = 65.5097
        avoid_div_zero = 0.000001

        #  input_row is a string containing two comma-separated values
        scaled_data1 = []
        scaled_data2 = []

        str_values = input_row.split(delimiter)
        str_values = [float(item) for item in str_values]

        #append scaled values
        scaled_data1.append((str_values[0] - min_val1) / (max_val1 - min_val1 + avoid_div_zero))
        scaled_data2.append((str_values[1] - min_val2) / (max_val2 - min_val2 + avoid_div_zero))

        # create NumPy arrays for the input
        X1 = np.array([scaled_data1])  # Note the use of square brackets inside the parentheses
        X2 = np.array([scaled_data2])

        # stack the two 1D arrays vertically
        input = np.vstack((X1, X2))
        #load the pre-trained biases and weights
        W1 = np.load("W1.npy")
        b1 = np.load("b1.npy")
        W2 = np.load("W2.npy")
        b2 = np.load("b2.npy")
        parameters= {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2

        }
        print("input:\n \n \n", input)
        output, weights = nn.forward_propagation(input_data=input,params=parameters)
       
        np.array(output)
        output = output.tolist()
        print("output: " ,output)
        print("output0: " ,output[0])
        print("output1: " ,output[1])
        output1 = (output[0][0])
        output1 = output1*7.99
        print("output1: " ,output1)
        output2 = (output[1][0])
        output2 = output2*7.938
        print("output2: " ,output2)
        return output2,output1
