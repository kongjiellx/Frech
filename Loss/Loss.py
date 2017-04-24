import numpy as np

class Loss(object):
    def __init__(self, name, y_, label):
        self.name = name
        self.y_ = y_
        self.label = label
        assert len(self.y_) == len(self.label)

    def gredient(self):
        if self.name == 'mae':
            grad = np.array([0.5 if i >= 0 else - 0.5 for i in self.y_ - self.label])
        elif self.name == 'square':
            grad = (1.0 / len(self.label)) * (self.y_ - self.label)
        elif self.name == 'categorical_cross_entropy':
            grad = self.y_ - self.label
        elif self.name == 'binary_cross_entropy':
            grad = self.y_ - self.label
        return grad

    def compute_loss(self):
        if self.name == 'mae':
            loss = np.sum(np.abs(self.y_ - self.label)) / len(self.label)
        elif self.name == 'square':
            loss = 1.0 / (2 * len(self.label)) * np.sum((self.y_ - self.label) * (self.y_ - self.label))
        elif self.name == 'categorical_cross_entropy':
            loss = - 1.0 * np.log(self.y_[np.argmax(self.label)] + 0.0001)
        elif self.name == 'binary_cross_entropy':
            loss = - 1.0 * ((1. - self.label) * np.log(1. - self.y_) + (self.label * np.log(self.y_)))
        return loss