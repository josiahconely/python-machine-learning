import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import pickle
from keras.callbacks import TensorBoard
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
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
from keras.models import model_from_yaml
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder


# load YAML and create model
print("Loading...")
yamlfile = open('model_SA.yaml', 'r')
model_yaml = yamlfile.read()
yamlfile.close()

model = model_from_yaml(model_yaml)
# load weights into new model
model.load_weights("model_SA.h5")
print("Loading...Done")



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

Phrase = "A lot of good things are happening. We are respected again throughout the world, and that's a great thing.@realDonaldTrump"
Phrase = [[Phrase]]
# pandas data frame Table with Columns and rows
max_df = pd.DataFrame(Phrase, index=range(0, 1, 1), columns=list('t'))

max_df['t'] = max_df['t'].apply(lambda x: x.lower())
max_df['t'] = max_df['t'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
print(max_df)

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(max_df['t'].values)
X = tokenizer.texts_to_sequences(max_df['t'].values)
X = pad_sequences(X, maxlen=28)

print("Model Input Vector")
print(X)
print("Prediction: " ,model.predict(X) )












