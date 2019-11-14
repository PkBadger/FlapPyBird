import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, input_number: int, hidden_number: int, output_number: int):
        self.input_nodes = input_number
        self.hidden_nodes = hidden_number
        self.output_nodes = output_number

        self.weights_ih = np.random.rand(hidden_number, input_number)
        self.weights_ho = np.random.rand(output_number, hidden_number)

        print(self.weights_ho)

        print(self.weights_ih)

        self.bias_h = np.random.rand(hidden_number)

        self.bias_o = np.random.rand(output_number)


    def feed_forward(self, input):
        hidden = np.dot(self.weights_ih, input)
        hidden = np.sum([hidden, self.bias_h], axis=0)

        print('hidden')
        print(hidden)
        
        afterSigmoid = sigmoid(hidden)
        print(afterSigmoid)

        output = np.dot(self.weights_ho, afterSigmoid)

        output = np.sum([output, self.bias_o], axis=0)

        return sigmoid(output)

    def train(self, inputs, targets):
        outputs = []
        for input in inputs:
            outputs.append(self.feed_forward(input))

        error = np.subtract(targets, outputs)
        weigths = np.transpose(self.weights_ho)

        hidden_errors = np.dot(weigths, error)

        print('TRAIN')
        print(error)
        print(weigths)
        print(hidden_errors)

if __name__ == '__main__':
    brain = NeuralNetwork(2,2,1)

    inputs = [[1,0]]

    targets = [1]

    output = brain.train(inputs, targets)

    print(output)
