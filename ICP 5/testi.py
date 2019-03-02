from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('College.csv',index_col=0)
x = df.drop('Private',axis=1)
sns.FacetGrid(df,hue="Private").map(plt.scatter,"Enroll","Expend")
plt.show()
km = KMeans(n_clusters=2)
km.fit(x)