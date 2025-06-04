import math


class Neuron:

    def __init__(self, value=0, gradient=0, lambda_value=0.9):
        self.value = value
        self.gradient = gradient
        self.lambda_value = lambda_value
        self.error = None

    def __str__(self) -> str:
        return f"value : {self.value} gradient : {self.gradient}"

    def __repr__(self):
        return f"value : {self.value} gradient : {self.gradient}"

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return self.value * other

    def activation_func(self, new_value):
        self.value = 1 / (1 + math.exp(-1 * self.lambda_value * new_value))

    def input_layer(self, new_value):
        self.value = new_value

    def calculate_error(self, target_value):
        self.error = target_value - self.value


    def set_error_for_hidden_layer(self, actual_error):
        self.error = actual_error

    def calculate_gradiant(self):
        self.gradient = self.lambda_value * self.value * (1 - self.value) * self.error
