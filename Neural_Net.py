import numpy as np

def main():
    Structure = [8,5,5,5,1]

    hidden_Layers_Amount = 3
    hidden_Layer_Node_Amount = 5
    Learning_Rate = 0.01

    weights_array = [] # list of 9*8, 9*9, ..., 9*1 matrices
    biases_array = [] # list of 9*1, 9*1, ..., 1*1 matrices
    fill_weights_and_biases(Structure, weights_array, biases_array)
    Outputs_array = [] # x*1 matrix
    print(weights_array)

    Neural_Net = np.array(intialize_Nueral_net(hidden_Layers_Amount, hidden_Layer_Node_Amount)) 

    # print("Weights array length: ", len(weights_array))
    # print("Biases array length: ", len(biases_array))
    # print("Neural net shape: ", Neural_Net.shape)
    
    trial_data = np.array([[107, 6.61, 6.28, 8, 0, 8, 8, 4]])
    # print("Trial data shape: ", trial_data.shape)
    
    transposed_data = trial_data.T
    # print("Transposed data shape: ", transposed_data.shape)
    
    feedfoward_output = feedfoward(transposed_data, weights_array, biases_array, Neural_Net)
    # print("Feedfoward output shape: ", feedfoward_output.shape)
    # print("Feedfoward output: ", feedfoward_output)

    trial_data_output = np.array([0])
    print("\n", Neural_Net)
    
    
    # cost_value = average_cost(feedfoward_output, trial_data_output_reshaped)
    # print("Average Cost value: ", cost_value)

    gradiant_w = np.array([]) 
    gradiant_b = np.array([]) 
    # backpropagation(gradiants_b, gradiants_w, biases_array, weights_array, cost_value)
    

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

        print("Weights shape: ", weights.shape)
        print("Biases shape: ", biases.shape)
 
 # feeds the input data through the neural network 
def feedfoward(input_data, weights_array, biases_array, Neural_Net):  
    output = input_data
    print (Neural_Net[0])

    for i in range(len(weights_array) - 1):
        output = sigmoid(np.dot(weights_array[i], output) + biases_array[i])
        print("\nLayer ", i, " output shape: ", output.shape)
        Neural_Net[i] = output.T[0]
    
    output = sigmoid(np.dot(weights_array[-1], output) + biases_array[-1])
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

def backpropagation(hidden_layers, Weights, Biases, Weights_gradients, Biases_gradients, outputs, actuals): # NOTE: Need to finish
    temp_Weights = Weights.T
    tnmp_biases = Biases.T
    
    reusable_math = sigmoid_derivative(outputs) * Loss_derivative(outputs, actuals)

    for i in range(len(temp_Weights)-1, -1, -1):
        Weights_gradients

    
    pass
    

if __name__ == "__main__":
    main()

