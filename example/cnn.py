from Layers.Dense import Dense
from Layers.Activation import Activation
from Layers.Convolutional import Convolutional
from Model.Model import Model
import numpy as np

input_data = np.arange(40).reshape((1, 2, 4, 5))
label = np.array([[1, 0, 0]])

conv = Convolutional(input_shape=(2, 4, 5), n_filters=1, filter_size=(2, 2))
res =  conv.forward(input_data[0])
print res