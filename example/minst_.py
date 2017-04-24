import mnist
import numpy as np

from Layers.Dense import Dense
from Layers.Activation import Activation
from Model.Model import Model

def one_hot(datas, nb_classes):
    ret = []
    for data in datas:
        d = np.zeros(shape=(nb_classes,))
        d[data] = 1
        ret.append(d)
    return np.array(ret)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

# x_test = x_test[0:100]
# y_test = y_test[0:100]
model = Model(loss_func='categorical_cross_entropy', optimizer_name='sgd_0.01')
model.add_layer(Dense(input_dim=784, out_dim=512))
model.add_layer(Activation(name='relu'))
model.add_layer(Dense(input_dim=512, out_dim=512))
model.add_layer(Activation(name='relu'))
model.add_layer(Dense(input_dim=512, out_dim=10))
model.add_layer(Activation(name='softmax'))
model.train(train_data=x_train, label=y_train, epochs=3)
model.predict(test_data=x_test, label=y_test)