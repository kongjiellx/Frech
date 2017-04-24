from Layers.Dense import Dense
from Layers.Activation import Activation
from Model.Model import Model
import numpy as np

input_data = np.array([[1, 2, 3]])
label = np.array([[1, 0, 0]])

model = Model(loss_func='categorical_cross_entropy', optimizer_name='sgd_0.11')
model.add_layer(Dense(input_dim=3, out_dim=3))
model.add_layer(Dense(input_dim=3, out_dim=3))
model.add_layer(Dense(input_dim=3, out_dim=3))
model.add_layer(Dense(input_dim=3, out_dim=3))
model.add_layer(Activation(name='softmax'))
model.train(train_data=input_data, label=label, epochs=10000)
print 'done'