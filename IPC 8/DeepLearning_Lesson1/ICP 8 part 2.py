import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation
# load dataset
import pandas as pd
from sklearn.model_selection import train_test_split








dataset = pd.read_csv("Breas Cancer.csv" )
# print(dataset)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(dataset['diagnosis'])
dataset['diagnosis']= le.transform(dataset['diagnosis'])
dataset = dataset.values

import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,2:31], dataset[:,1],
                                                    test_size=0.25, random_state=87)
#np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(40, input_dim=29, activation='relu')) # hidden layer
my_first_nn.add(Dense(40, activation='sigmoid')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.metrics_names)
print(my_first_nn.evaluate(X_test, Y_test, verbose=0))


