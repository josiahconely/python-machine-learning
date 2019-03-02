import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats

#part 1
#gets the housing data
df = pd.read_csv('train.csv')

#extracts the two columns of interest for display
X = df['GarageArea']
Y = df['SalePrice']
#displays the pre cleaned data
plt.show(sns.scatterplot(X,Y))

#Exptracts the two columns of interest for cleaning
xy = df[['SalePrice','GarageArea']]
#This line removes all rows of the data that have an element that is more than
# 3 standard deviations away from the mean of the coloumn
xy_cleaned = xy[(np.abs(stats.zscore(xy)) < 3).all(axis =1)]

X_cleaned = xy_cleaned['GarageArea']
Y_cleaned = xy_cleaned['SalePrice']

plt.show(sns.scatterplot(X_cleaned,Y_cleaned))


#############
#Part 2

dfWeather =pd.read_csv('weatherHistory.csv')

# this code normalized the Summary column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(dfWeather['Summary'])
dfWeather['Summary']= le.transform(dfWeather['Summary'])

#selects training variables
X = dfWeather [['Summary','Humidity', 'Wind Speed (km/h)','Visibility (km)', 'Pressure (millibars)']]
Y = dfWeather['Temperature (C)']
#removes any nulls
X.fillna(X.mean())
Y.fillna(Y.mean())

#splits the training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.4, random_state= 101)


lm = LinearRegression()
lm.fit(X_train,Y_train)
# shows a score of the fit
print(lm.score(X_test,Y_test))
from sklearn.metrics import mean_squared_error
prediction =lm.predict(X_test)
print ('rmse',mean_squared_error(Y_test, prediction))
