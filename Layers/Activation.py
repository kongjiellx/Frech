import numpy as np
from Layer import Layer

class Activation(Layer):
    def __init__(self, name):
        super(Activation, self).__init__()
        self.name = name
        pass

    def forward(self, data):
        self.data = data
        if self.name == 'sigmoid':
            self.out = 1. / (1. + np.exp(-1. * data))
        elif self.name == 'softmax':
            self.out = np.exp(data) / np.sum(np.exp(data))
        elif self.name == 'relu':
            self.out = np.array([i if i > 0 else 0 for i in data])
        return self.out

    def bp(self, delta):
        if self.name == 'sigmoid':
            next_delta = delta * self.out * (1. - self.out)
        elif self.name == 'softmax':
            next_delta = delta * self.out * (1. - self.out)
        elif self.name == 'relu':
            next_delta = delta * np.array([1 if i > 0 else 0 for i in self.data])
        return next_delta

    def updata_params(self, optimizer):
        pass