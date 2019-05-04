import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import pickle
from keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('spam.csv',encoding = "ISO-8859-1")
# Keeping only the neccessary columns
data = data[['v2','v1']]



data['v2'] = data['v2'].apply(lambda x: x.lower())
data['v2'] = data['v2'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

print(data[data["v1"] == 'spam'].size)
print(data[data['v1'] == 'ham'].size)

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['v2'].values)
X = tokenizer.texts_to_sequences(data['v2'].values)
print(X)
X = pad_sequences(X)
print(X)
embed_dim = 128
lstm_out = 196
def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model
# print(model.summary())

labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['v1'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
model = createmodel()


tensorboard = TensorBoard(log_dir='./SA_logsPart3', histogram_freq=0,
                          write_graph=True, write_images=False)
batch_size = 32
hist = model.fit(X_train, Y_train,
                 epochs=5,
                 batch_size=batch_size,
                 verbose=2,
                 callbacks=[tensorboard])

score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)

#
# # serialize model to YAML
# model_yaml = model.to_yaml()
# with open("model_SA.yaml", "w") as yaml_file:
#     yaml_file.write(model_yaml)
# # serialize weights to HDF5
# model.save_weights("model_SA.h5")
#
# # save history:
# model_ori = open('model_SA.pckl', 'wb')
# pickle.dump(hist.history, model_ori)
# model_ori.close()