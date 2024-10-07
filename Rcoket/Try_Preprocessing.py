import numpy as np
import pandas as pd


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
        self.scaled_train_data = None
        self.scaled_validation_data = None
        self.scaled_test_data = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None

    def read_csv(self):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.file_path)

        # Get the last two columns
        last_two_columns = df.iloc[:, -2:]

        # Switch the positions of the last two columns
        df.iloc[:, -2:] = df.iloc[:, [-1, -2]].values
        self.df = df
        # return df

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



