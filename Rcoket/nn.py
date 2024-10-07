import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Preprocessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.column_names = ['X_distance_to_variable', 'Y_distance_to_variable', 'X_velocity', 'Y_velocity']
        self.df = None
        self.min_values_train = None
        self.max_values_train = None
        self.min_values_validation = None
        self.max_values_validation = None
        self.min_values_test = None
        self.max_values_test = None

    def read_csv(self):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.file_path)

        # Get the last two columns
        last_two_columns = df.iloc[:, -2:]

        # Switch the positions of the last two columns
        df.iloc[:, -2:] = df.iloc[:, [-1, -2]].values
        self.df = df
        return df

    def randomize_data(self):
        # Randomize the data
        # frac=1 to shuffle the entire data_set
        # re-setting the index at the end.
        self.df = self.df.sample(frac=1, random_state=365).reset_index(drop=True)

    def split_data(self):
        # Split the data into training, validation, and testing sets (70%, 15%, 15%)
        total_rows = self.df.shape[0]
        # use int to make sure the number of rows is a whole number
        train_end = int(0.7 * total_rows)
        validation_end = int(0.85 * total_rows)

        # create the training data from the start to the train_end -1
        self.train_data = self.df.iloc[:train_end, :]
        # create the validation data from the end of training till the end of validation_end-1
        self.validation_data = self.df.iloc[train_end:validation_end, :]
        # The rest is for the test set
        self.test_data = self.df.iloc[validation_end:, :]

    def min_max_scaling(self):
        # Calculate and save minimum and maximum values for each feature in the training set
        self.min_values_train = self.train_data.min()
        self.max_values_train = self.train_data.max()
        self.min_values_validation = self.validation_data.min()
        self.max_values_validation = self.validation_data.max()
        self.min_values_test = self.test_data.min()
        self.max_values_test = self.test_data.max()

        # avoid division by zeros for floats
        # very unlikely to get zero after adding this number more so than leaving the division as is
        avoid_div_by_zero = 1e-10
        # Apply Min-Max scaling to the training set using train set's min and max
        self.scaled_train_data = (self.train_data - self.min_values_train) / (
                self.max_values_train - self.min_values_train + avoid_div_by_zero)
        # Apply Min-Max scaling to the validation set using validation set's min and max
        self.scaled_validation_data = (self.validation_data - self.min_values_validation) / (
                self.max_values_validation - self.min_values_validation + avoid_div_by_zero)
        # Apply Min-Max scaling to the testing set using testing set's min and max
        self.scaled_test_data = (self.test_data - self.min_values_test) / (
                self.max_values_test - self.min_values_test + avoid_div_by_zero)
        # Return the scaled datasets
        return self.scaled_train_data, self.scaled_validation_data, self.scaled_test_data

    def min_max_inverse(self, scaled_data, data_type='train'):

        if data_type == 'train':
            min_values = self.min_values_train
            max_values = self.max_values_train
        elif data_type == 'validation':
            min_values = self.min_values_validation
            max_values = self.max_values_validation
        elif data_type == 'test':
            min_values = self.min_values_test
            max_values = self.max_values_test
        else:
            print('error in reverse scaling')

        # Inverse scaling formula
        avoid_div_by_zero = 1e-10
        original_data = scaled_data * (max_values - min_values + avoid_div_by_zero) + min_values

        return original_data

    def switch_all_columns_to_rows(self):

        # Melt the DataFrame to switch all columns into rows
        df_melted = pd.melt(self.df)

        return df_melted


def sigmoid(x, lamda=0.7):
    return 1 / (1 + np.exp(-x * lamda))


# Sigmoid derivative
def sigmoid_derivative(x, lamda=0.7):
    return lamda * sigmoid(x) * (1 - sigmoid(x))


# def layer(size_of_input, size_of_output, size_of_hidden_layer):
#     size_of_input = size_of_input.shape
#     size_of_output = size_of_output
#     size_of_hidden_layer = size_of_hidden_layer
#     return size_of_input, size_of_output, size_of_hidden_layer


def forward_propagation(input_data, params):
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    # Note: change the name of the variables as the don't represent the actual thing
    output_of_input = np.dot(W1, input_data) + b1
    input_to_hidden_layer = sigmoid(output_of_input)
    output_of_hidden_layer = np.dot(W2, input_to_hidden_layer) + b2
    input_of_output_layer = sigmoid(output_of_hidden_layer)
    saved_dict = {"Z1": output_of_input,
                  "A1": input_to_hidden_layer,
                  "Z2": output_of_hidden_layer,
                  "A2": input_of_output_layer

                  }
    return input_of_output_layer, saved_dict


def cost_function(A2, y_actual):
    difference_squared = (y_actual - A2) ** 2

    # Calculate mean squared error
    MSE = np.mean(difference_squared)

    # Calculate the square root of MSE
    RMSE = np.sqrt(MSE)

    return RMSE


def back_propagation(parameters, saved, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    A1 = saved["A1"]
    A2 = saved["A2"]
    Z1 = saved["Z1"]
    Z2 = saved["Z2"]
    # derivatives
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * (np.sum(dZ2, axis=1, keepdims=True))
    dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(Z1)
    dW1 = (1 / m) * (np.dot(dZ1, X.T))
    db1 = (1 / m) * (np.sum(dZ1, axis=1, keepdims=True))
    # gradients of each step
    gradients = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2,
                 "dZ1": dZ1,
                 "dZ2": dZ2
                 }

    return gradients


def update_weights_and_biases(parameters, grads, learning_rate=0.001):
    # retrieve current params
    # print(parameters)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # retrieve current grads
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    # make the new params (update)
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    # return new params
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    # print(parameters)
    return parameters


def params(size_of_input, size_of_output, size_of_hidden_layer):
    # print(size_of_hidden_layer)
    weights12 = np.random.randn(size_of_hidden_layer, size_of_input) * 0.01
    bias12 = np.zeros((size_of_hidden_layer, 1))
    weights23 = np.random.randn(size_of_output, size_of_hidden_layer) * 0.01
    bias23 = np.zeros((size_of_output, 1))
    parameters = {"W1": weights12,
                  "b1": bias12,
                  "W2": weights23,
                  "b2": bias23}
    return parameters





def Neural_Network(X_train, Y_train, X_val, Y_val, number_of_hidden_layer_neurons=2, epochs=100, batch_size=25):
    np.random.seed(54)
    size_of_input = X_train.shape[0]
    size_of_output = Y_train.shape[0]
    parameters = params(size_of_input, size_of_output, number_of_hidden_layer_neurons)

    training_costs = []  # Store training costs for each iteration
    validation_costs = []  # Store validation costs for each iteration

    for i in range(epochs):
        total_train_cost = 0
        total_val_cost = 0

        # Training batches
        for j in range(0, X_train.shape[1], batch_size):
            X_batch = X_train[:, j:j + batch_size]
            Y_batch = Y_train[:, j:j + batch_size]
            A2, stored = forward_propagation(X_batch, parameters)
            grads = back_propagation(parameters, stored, X_batch, Y_batch)
            parameters = update_weights_and_biases(parameters, grads)
            total_train_cost += cost_function(A2, Y_batch)

        avg_train_cost = total_train_cost / (X_train.shape[1] / batch_size)
        training_costs.append(avg_train_cost)

        # Validation batches
        for q in range(0, X_val.shape[1], batch_size):
            X_batch_val = X_val[:, q:q + batch_size]
            Y_batch_val = Y_val[:, q:q + batch_size]
            A2_val, _ = forward_propagation(X_batch_val, parameters)
            total_val_cost += cost_function(A2_val, Y_batch_val)

        avg_val_cost = total_val_cost / (X_val.shape[1] / batch_size)
        validation_costs.append(avg_val_cost)

        print(f"Epoch {i + 1}/{epochs}, Training Cost: {avg_train_cost:.4f}, Validation Cost: {avg_val_cost:.4f}")

    # Plot the training and validation costs
    plt.plot(range(epochs), training_costs, label='Training Cost')
    plt.plot(range(epochs), validation_costs, label='Validation Cost')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()

    return parameters

"""
file_name = "ce889_dataCollection.csv"
file_name = Preprocessing(file_name)
file_name.read_csv()
df = file_name.df
# print("read_file \n", df.head)
file_name.randomize_data()
random = file_name.df
# print("randomized \n", random.head)
file_name.split_data()
file_name.min_max_scaling()
scaled_train = file_name.scaled_train_data
scaled_validation = file_name.scaled_validation_data
# scaled data to numpy
numpy_of_scaled_data = scaled_train.to_numpy()
numpy_of_scaled_validation = scaled_validation.to_numpy()
# transpose of scaled data
transpose_of_training = numpy_of_scaled_data.T
transpose_of_validation = numpy_of_scaled_validation.T

print(transpose_of_training.shape)

# Determine the new shape (y/2)
new_y = numpy_of_scaled_data.shape[1] // 2

# Split the array into two separate arrays
X_train = transpose_of_training[0:2, :].copy()  # Use .copy() to create a new array
X_val = transpose_of_validation[0:2, :].copy()
Y_train = transpose_of_training[2:4, :].copy()  # Use .copy() to create a new array
Y_val = transpose_of_validation[2:4, :].copy()
W1, b1, W2, b2 = Neural_Network(X_train, Y_train, X_val, Y_val)
# print(f"W1:{W1}\n\n\nb1:{b1}\n\n\nW2:{W2}\n\n\nb2:{b2}\n\n\n")
"""