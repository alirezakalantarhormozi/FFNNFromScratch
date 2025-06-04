from Neuron import Neuron
import numpy as np


class Layer:

    def __init__(self, number_of_neuron, level, is_input_layer=False, is_output_layer=False):
        self.number_of_neuron = number_of_neuron
        self.is_input_layer = is_input_layer
        self.is_output_layer = is_output_layer
        self.Neurons = np.array([])
        self.level = level
        for number in range(number_of_neuron):
            self.Neurons = np.append(self.Neurons, Neuron())

    def __str__(self) -> str:
        s = ""
        for neuron in self.Neurons:
            s += "Level : "+str(self.level) +" [ " + neuron.__str__() + " ]" + "\n"
        return s

    def __repr__(self):
        s = ""
        for neuron in self.Neurons:
            s += "Level : "+ str(self.level) +" [ " + neuron.__str__() + " ]" + "\n"
        return s

    def update_values(self, vector_of_new_values):
        # if it is input layer dont need to call activation func just adjust numeric value to the neuron
        if self.is_input_layer:
            for index, neuron in enumerate(self.Neurons):
                neuron.input_layer(vector_of_new_values[index])
        elif self.is_output_layer:
            for index, neuron in enumerate(self.Neurons):
                neuron.activation_func(vector_of_new_values[index])
        else:
            for index, neuron in enumerate(self.Neurons):
                if index == 0:
                    neuron.input_layer(1) # bias always the first input is 1 for each layer
                else:
                    neuron.activation_func(vector_of_new_values[index])

    def calculate_or_set_error_and_calculate_grad(self, target_or_actual_error):
        if self.is_output_layer:
            for index, neuron in enumerate(self.Neurons):
                neuron.calculate_error(target_or_actual_error[index])
                neuron.calculate_gradiant()
        else:
            for index, neuron in enumerate(self.Neurons):
                neuron.set_error_for_hidden_layer(target_or_actual_error[index])
                neuron.calculate_gradiant()

    def get_gradiant_vector(self):
        temp_list = []
        for neuron in self.Neurons:
            temp_list.append(neuron.gradient)
        return np.array(temp_list)