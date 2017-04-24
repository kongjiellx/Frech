import numpy as np

class Layer(object):
    def __init__(self):
        self.params = []

    def forward(self, data):
        raise NotImplementedError

    def bp(self, delta):
        raise NotImplementedError

    def updata_params(self, optimizer):
        raise NotImplementedError

