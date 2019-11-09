import numpy as np

#sigmoid function used to return an value positive value


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# to find the difference using slop we take the derivative of the sigmoid
def sigmoid_derivative(x):
    return x*(1-x)


# we have an training input
input_training = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

print(input_training)


# we have training output
output_training = np.array([[0, 1, 1, 0]]).T


np.random.seed(1)


# the formula generates a array of random numbers between -1 to 1 in a array of 3
syn_weight = 2*np.random.random((3, 1))-1
# the do product of the input_training and the output

for i in range(10000):
    input_layer = input_training
# we send the dot product of syn weight and input to signoid which is the will be aour pedicted input

    output_layer = sigmoid(np.dot(input_layer, syn_weight))

    error = output_training - output_layer

# to adjust the synaptics weights we take the derivative of the outputlayer and multiple it with error
    adjustment = error * sigmoid_derivative(output_layer)

    syn_weight += np.dot(input_layer.T, adjustment)

    new = np.array([[1, 0, 0]])

print(sigmoid(np.dot(new, syn_weight)))