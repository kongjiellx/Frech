from Layer import Layer
import numpy as np

class Convolutional(Layer):
    '''
    input_shape: 3d_tensor [channels, width, height]
    filter_size: 2d_tensor [width_len, height_len]
    '''

    def __init__(self, input_shape, n_filters, filter_size, stride=1):
        super(Convolutional, self).__init__()
        assert len(filter_size) == 2
        self.filter_size = filter_size
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.stride = stride
        # self.W = np.ones(shape=(n_filters, filter_size[0], filter_size[1]))
        self.W = np.random.uniform(low=-0.1, high=0.1, size=(n_filters, filter_size[0], filter_size[1]))
        self.b = np.zeros(n_filters)

    def forward(self, data):
        self.data = data
        assert data.shape == self.input_shape
        feature_maps = []
        for n in range(self.n_filters):
            feature_map = []
            for i in range(self.input_shape[1] - self.filter_size[0] + 1):
                col = []
                for j in range(self.input_shape[2] - self.filter_size[1] + 1):
                    point = np.sum(self.W[n] * data[:, i: i+self.filter_size[0], j: j+self.filter_size[1]]) + self.b[n]
                    col.append(point)
                feature_map.append(col)
            feature_maps.append(feature_map)
        self.out = np.array(feature_maps)
        return self.out

    def bp(self, delta):
        next_delta = np.dot(delta, self.params[0].T)
        g = []
        for d in delta:
            g.append((d * self.data).reshape((self.input_dim, 1)))
        self.grad_W = np.concatenate(g, axis=1)
        self.grad_b = delta * 1
        self.grads = [self.grad_W, self.grad_b]
        return next_delta

    def updata_params(self, optimizer):
        assert len(self.params) == len(self.grads)
        for i in range(len(self.params)):
            self.params[i] = optimizer.get_updates(self.params[i], self.grads[i])

