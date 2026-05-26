import numpy as np
import csv
import tkinter as tk
from tkinter import filedialog
import pandas as pd


# sqrt(2/n) n is input node amount

def main():
    Structure = [8,5,5,5,1]

    hidden_Layers_Amount = 3
    hidden_Layer_Node_Amount = 5
    Learning_Rate = 0.01

    weights_array = [] # list of 9*8, 9*9, ..., 9*1 matrices
    biases_array = [] # list of 9*1, 9*1, ..., 1*1 matrices
    fill_weights_and_biases(Structure, weights_array, biases_array)
    Pre_activations_array = [] # list of 9*1, 9*1, ..., 1*1 matrices
    Post_activations_array = [] # list of 9*1, 9*1, ..., 1*1 matrices
    Outputs_array = [] # x*1 matrix
    
    trial_data = np.array([[107, 6.61, 6.28, 8, 0, 8, 8, 4]])
    transposed_data = trial_data.T
    print(load_data())
    
    feedfoward_output = feedfoward(transposed_data, weights_array, biases_array, Pre_activations_array, Post_activations_array)
    trial_data_output = np.array([0])
    

    # cost_value = average_cost(feedfoward_output, trial_data_output_reshaped)
    # print("Average Cost value: ", cost_value)

    gradiant_w = [0] * len(weights_array)
    gradiant_b =  [0] * len(biases_array)   
    backpropagation(Pre_activations_array, Post_activations_array, weights_array, biases_array, gradiant_w, gradiant_b, feedfoward_output, trial_data_output)
    update_weights_and_biases(weights_array, biases_array, gradiant_w, gradiant_b, Learning_Rate)

def load_data():
    root = tk.Tk()
    root.withdraw()

    # Show the 'Open' dialog box and return the selected file path
    file_name = filedialog.askopenfilename()

    with open(file_name, 'r', newline='') as raw_data:
        data_list = pd.read_csv(raw_data).to_numpy()

    return data_list

    

def update_weights_and_biases(weights_array, biases_array, gradiant_w, gradiant_b, learning_rate):
    for i in range(len(weights_array)):
        weights_array[i] -= learning_rate * gradiant_w[i]
        biases_array[i] -= learning_rate * gradiant_b[i]

    
    

# initializes neural net nodes to 0 and returns a list
def intialize_Nueral_net(layers_amount, layer_node_amount):
    Neural_Net = []

    for i in range(layers_amount):
        layer = [0] * layer_node_amount
        Neural_Net.append(layer)

    return Neural_Net

# fills weights and biases arrays with random values based on the layer node amounts
def fill_weights_and_biases(Layer_Node_Amounts, weights_array, biases_array):
    
    for i in range(len(Layer_Node_Amounts)-1):
        weights = np.random.randn(Layer_Node_Amounts[i+1], Layer_Node_Amounts[i])
        biases = np.random.randn(Layer_Node_Amounts[i+1], 1)
        

        weights_array.append(weights)
        biases_array.append(biases)

        
 
 # feeds the input data through the neural network 
def feedfoward(input_data, weights_array, biases_array, Pre_activations_array, Post_activations_array):
    output = input_data
    Post_activations_array.append(input_data)  # did this to make indexing easier

    for i in range(len(weights_array)):
        Pre_activations_output = np.dot(weights_array[i], output) + biases_array[i]
        Pre_activations_array.append(Pre_activations_output)

        output = sigmoid(Pre_activations_output)
        Post_activations_array.append(output)

    print("\n Final output: ", output[0][0])
    return output

# activation function that maps any real-valued number into the (0, 1) interval
def sigmoid(arr):    
    return 1 / (1 + np.exp(-arr))

def sigmoid_derivative(arr):
    return sigmoid(arr) * (1 - sigmoid(arr))

def average_cost(outputs, expected_outputs): 
    loss_values = []

    for i in range(len(outputs)):
        loss = Loss_function(outputs[i], expected_outputs[i])
        loss_values.append(loss)
    
    return np.mean(loss_values)

def Loss_function(output, actual):
    return -(actual * np.log(output) + (1 - actual) * np.log(1 - output))

def Loss_derivative(output, actual):
    return (-actual / output) + ((1 - actual) / (1 - output))

def backpropagation(pre_activations, post_activations, Weights, Biases, Weights_gradients, Biases_gradients, outputs, actuals):
    # Compute output layer delta
    delta = sigmoid_derivative(pre_activations[-1]) * Loss_derivative(outputs, actuals)
    print("\n pre activations: ", pre_activations)
    print("\n post activations: ", post_activations)

    for i in range(len(Weights) - 1, -1, -1):
        # post_activations[i] is the activation feeding INTO layer i
        Weights_gradients[i] = np.dot(delta, post_activations[i].T)
        Biases_gradients[i] = delta
        print("\n pre_activations[i] in use: ", pre_activations[i])
        print("\n post_activations[i] in use: ", post_activations[i])
        print("\n delta in use: ", delta)

        if i > 0:
            # Propagate delta to the previous layer
            delta = np.dot(Weights[i].T, delta) * sigmoid_derivative(pre_activations[i - 1])
    

if __name__ == "__main__":
    main()

