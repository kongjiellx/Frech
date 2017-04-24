class Optimizer(object):
    def __init__(self, name):
        self.name = name

    def get_updates(self, param, grad):
        if self.name.split('_')[0] == 'sgd':
            return self.sgd(param, grad, float(self.name.split('_')[1]))

    def sgd(self, param, grad, lr=0.5):
        new_param = param - lr * grad
        return new_param
