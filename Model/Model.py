from Loss.Loss import Loss
from Optimizer.Optimizer import Optimizer
import numpy as np

class Model(object):
    def __init__(self, loss_func, optimizer_name):
        self.layers = []
        self.loss_name = loss_func
        self.optimizer = Optimizer(optimizer_name)

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, train_data):
        data = train_data
        for layer in self.layers:
            data = layer.forward(data)
        return data

    def bp(self, delta):
        for layer in self.layers[::-1]:
            delta = layer.bp(delta)
            layer.updata_params(self.optimizer)

    def train(self, train_data, label, epochs):
        for i in range(epochs):
            total_loss = 0.
            for j in range(len(train_data)):
                y_ = self.forward(train_data[j])
                # print y_
                loss = Loss(self.loss_name, y_, label[j])
                loss_value = loss.compute_loss()
                total_loss += loss_value
                delta = loss_value * loss.gredient()
                self.bp(delta)
            print total_loss

    def predict(self, test_data, label):
        right = 0
        for i in range(len(test_data)):
            y_ = self.forward(test_data[i])
            if label[i][np.argmax(y_)] == 1:
                right += 1
        print 'acc: ', right * 1.0 / (len(label))
