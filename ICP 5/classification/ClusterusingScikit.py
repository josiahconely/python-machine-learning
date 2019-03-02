from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('classification/train.csv')
# see how many samples we have of each species
print(dataset["Age"].value_counts())

sns.FacetGrid(dataset, hue="Age", size=4).map(plt.scatter, "Age", "Fare")
plt.show()
sns.FacetGrid(dataset, hue="Age", size=4).map(plt.scatter, "Age", "SibSp")
plt.show()
sns.FacetGrid(dataset, hue="Age", size=4).map(plt.scatter, "Age", "Parch")
plt.show()