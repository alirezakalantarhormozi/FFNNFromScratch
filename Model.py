import numpy as np
from Layer import Layer
import math
import sys
from time import sleep
import matplotlib.pylab as plt


class Model:
    # Initializing Model class
    def __init__(self, number_of_input, number_of_output, list_of_number_of_hidden_layer_neuron, number_of_epoch, eta, alpha):
        self.number_of_input = number_of_input
        self.number_of_output = number_of_output
        self.list_of_number_of_hidden_layer_neuron = list_of_number_of_hidden_layer_neuron
        self.number_of_epoch = number_of_epoch
        self.eta = eta
        self.train_data = None
        self.train_target = None
        self.loss = []
        self.validation_loss = []

        self.alpha = alpha
        # make a list of dimensions for wight matrices
        print(list_of_number_of_hidden_layer_neuron)
        list_of_number_of_hidden_layer_neuron = [i + 1 for i in list_of_number_of_hidden_layer_neuron ]
        print(list_of_number_of_hidden_layer_neuron)
        list_of_matrices_dimensions = list_of_number_of_hidden_layer_neuron.copy()
        list_of_matrices_dimensions.insert(0, number_of_input + 1)
        list_of_matrices_dimensions.append(number_of_output)
        # using dimensions to built wight matrices
        temp_list_wight_matrices = []
        for index in range(len(list_of_matrices_dimensions)-1):
            temp = np.random.rand(list_of_matrices_dimensions[index + 1], list_of_matrices_dimensions[index])
            temp_list_wight_matrices.append(temp)

        self.wights = np.array(temp_list_wight_matrices, dtype=object)
        self.previous_delta_weights = np.empty((self.wights.shape[0],), dtype=object)
        # print(self.previous_delta_weights.shape)
        # print(self.previous_delta_weights)
        # print(self.wights)
        # self.wights = np.round(self.wights, 1)
        # built layers
        temp_list_layer = []
        for index, number_of_each_layer in enumerate(list_of_matrices_dimensions):
            # input layer
            if index == 0:
                temp_list_layer.append(Layer(number_of_each_layer, index, True, False))
            # output layer
            elif index == len(list_of_matrices_dimensions) - 1:
                temp_list_layer.append(Layer(number_of_each_layer, index, False, True))
            # hidden layer
            else:
                temp_list_layer.append(Layer(number_of_each_layer, index))
        self.layers = np.array(temp_list_layer)

    def fit(self, train_data, train_target, validation_data, validation_target):
        self.train_data = train_data
        self.train_target = train_target
        for epoch in range(self.number_of_epoch):
            # print(epoch)
            su = 0
            for i in range(len(self.train_data)):
                if i % ((len(self.train_data)/100) * 1) == 0:  # each 5 percent  go for printing
                    temp = '['
                    for z in range(int(i // ((len(self.train_data) / 100) * 1))):  # print = for each 5 percent
                        temp += '='
                    temp += '>'
                    # fill the rest with blank space
                    for z in range(int((len(self.train_data) - i) // ((len(self.train_data) / 100) * 1))):
                        temp += ' '
                    temp += ']'
                    # come in the start of line (
                    # it give us the ability to overwrite the line
                    # also make it like progress bar
                    sys.stdout.write('\r')
                    sys.stdout.write(temp)  # print what you want
                    sys.stdout.flush()  # flush
                self.layers[0].update_values(np.append([1], train_data[i])) # append is for bias 1
                self.feed_forward()
                # calculate Errors and Gradiant
                self.calculate_gradiant(i, train_target)
                # calculate for MSE
                for neuron in self.layers[-1].Neurons:
                    su += math.pow(neuron.error, 2)
                # back propagation
                self.back_propagation()
            los = math.sqrt((1 / (len(train_data) * self.number_of_output)) * su)
            # print(los)
            self.loss.append(los)
            validation_los = self.validation_RMSE(validation_data, validation_target)
            self.validation_loss.append(validation_los)
            print(f"Epoch : {epoch + 1}  |  Training Loss : {los}  |  Validation Loss : {validation_los}   ")
        print(self.number_of_epoch)
        print(self.loss)
        plt.plot(range(1, self.number_of_epoch+1), self.loss, 'g', label='Training loss')
        plt.plot(range(1, self.number_of_epoch+1), self.validation_loss, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


    def feed_forward(self):
        for index in range(len(self.layers)-1):
            temp = np.dot(self.wights[index], self.layers[index].Neurons)
            self.layers[index+1].update_values(temp)

    def back_propagation(self):
        for index, wight in enumerate(self.wights):
            gradiant_vector = self.layers[index+1].get_gradiant_vector()
            # print(gradiant_vector)
            # print(wight.shape[1])
            matrix_for_gradiant_in_shape_of_wight = np.vstack([gradiant_vector] * wight.shape[1])
            matrix_for_gradiant_in_shape_of_wight = matrix_for_gradiant_in_shape_of_wight.T
            # print(matrix_for_gradiant_in_shape_of_wight)
            matrix_for_value_in_shape_of_wight = np.vstack([self.layers[index].Neurons] * wight.shape[0])
            matrix_for_value_in_shape_of_wight = matrix_for_value_in_shape_of_wight
            # print(matrix_for_value_in_shape_of_wight)

            delta_wight = np.multiply(matrix_for_value_in_shape_of_wight,matrix_for_gradiant_in_shape_of_wight)
            # print(delta_wight)
            # delta_wight = delta_wight * 1 * self.eta
            if self.previous_delta_weights[index] is None:
                delta_wight = delta_wight * 1 * self.eta
                self.previous_delta_weights[index] = delta_wight
            else:
                delta_wight = (delta_wight * 1 * self.eta) + (self.alpha * self.previous_delta_weights[index])
                self.previous_delta_weights[index] = delta_wight
            # print(delta_wight)
            # print('previous: ', wight)
            self.wights[index] = wight + delta_wight
            # print('new : ', wight)
            # delta_wight = np.dot(gradiant_vector)

    def calculate_gradiant(self, index_of_data_target, target):
        for index in range(len(self.layers) - 1, 0, -1):
            if index == len(self.layers) - 1:
                self.layers[index].calculate_or_set_error_and_calculate_grad(target[index_of_data_target])
            else:
                gradiant_vector = self.layers[index+1].get_gradiant_vector()
                temp = np.dot(gradiant_vector, self.wights[index])
                self.layers[index].calculate_or_set_error_and_calculate_grad(temp)

    def predict(self, test_data, test_target):
        su = 0
        temp_list = []
        for i in range(len(test_data)):
            self.layers[0].update_values(np.append([1], test_data[i]))
            self.feed_forward()
            # calculate Errors and Gradiant
            self.calculate_gradiant(i, test_target)
            # calculate for MSE
            for neuron in self.layers[-1].Neurons:
                temp_list.append(neuron.value)
                su += math.pow(neuron.error, 2)
        result = np.array(temp_list)
        los = math.sqrt((1 / (len(test_data) * self.number_of_output)) * su)
        print(los)
        return result

    def validation_RMSE(self, test_data, test_target):
        su = 0
        temp_list = []
        for i in range(len(test_data)):
            self.layers[0].update_values(np.append([1], test_data[i]))
            self.feed_forward()
            # calculate Errors and Gradiant
            self.calculate_gradiant(i, test_target)
            # calculate for MSE
            for neuron in self.layers[-1].Neurons:
                temp_list.append(neuron.value)
                su += math.pow(neuron.error, 2)
        result = np.array(temp_list)
        los = math.sqrt((1 / (len(test_data) * self.number_of_output)) * su)
        return los

    def __str__(self) -> str:
        print(self.layers)
        print(self.wights)
        return ""

    def __repr__(self):
        print(self.layers)
        print(self.wights)
        return ""