from Layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_dim, out_dim):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.W = np.random.uniform(low=-0.1, high=0.1, size=(input_dim, out_dim))
        self.b = np.zeros(out_dim)

    def forward(self, data):
        self.data = data
        self.out = np.dot(self.data, self.W) + self.b
        return self.out

    def bp(self, delta):
        next_delta = np.dot(delta, self.W.T)
        g = []
        for d in delta:
            g.append((d * self.data).reshape((self.input_dim, 1)))
        self.grad_W = np.concatenate(g, axis=1)
        self.grad_b = delta * 1
        return next_delta

    def updata_params(self, optimizer):
        self.W = optimizer.get_updates(self.W, self.grad_W)
        self.b = optimizer.get_updates(self.b, self.grad_b)
