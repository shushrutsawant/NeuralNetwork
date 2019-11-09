import numpy as np

class Neural:

    def __init__(self):
        self.weight = 0
        np.random.seed(1)

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self,x):
        return x*(1-x)

    def train(self, input_data, output, size , learning):

        np.random.seed(1)

        self.weight = 2*np.random.random((size, 1))-1

        output_temp = output.T

        for i in range(learning):

            input_layer = input_data

            output_layer = self.sigmoid(np.dot(input_layer, self.weight))

            error = output_temp - output_layer

            adjustment = error * self.sigmoid_derivative(output_layer)

            self.weight += np.dot(input_data.T, adjustment)

    def predict(self,x):
        print(self.sigmoid(np.dot(x, self.weight)))