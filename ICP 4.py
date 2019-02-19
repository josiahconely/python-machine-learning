import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


#creates pandas serise
iris = pd.read_csv('iris.csv')
attributes = iris[['sepal length','sepal width','petal length','petal width']]
classes = iris['class']
#print (iris)
#print (attributes)
#print (classes)

#splits the serise into test and train sets
attributes_train, attributes_test, class_train, class_test = train_test_split(attributes, classes, test_size=0.5)

#sets up Naive Bayes distribution for the training data
gnb = GaussianNB()
gnb.fit(attributes_train, class_train)
#runs the test on the test portion of the data
print("given the training data, the test resulted with {}".format(gnb.score(attributes_test, class_test)))


####################
##part 2############


# creates the sv
svc = LinearSVC(max_iter=20000)
svc.fit(attributes_train, class_train)
#runs the test on the test data
print("using a linear smv model,given the training data, the test resulted with {}".format(svc.score(attributes_test, class_test)))

# AFTER RUNNING BOTH TESTS NAIVE BAYES PERFORMED BETTER
#
#NAIVE BAYES = 0.9733333333333334
#SVM         = 0.9466666666666667

# I believe it was better because of the size of the data sets. If it were larger, perhaps svm would have
#preformed better


################
##Part 3 #######

svc2 = SVC(kernel= 'rbf')
svc2.fit(attributes_train, class_train)
print("using a 'rbf' svc model,given the training data, the test resulted with {}".format(svc2.score(attributes_test, class_test)))

# running this program several times gave different results on every run of the program
# for this reason I cannot say with certainty what the result is
#however, based on this test result, rbf seemed to produce the best result
#given the training data, the test resulted with 0.9733333333333334
#using a linear smv model,given the training data, the test resulted with 0.9733333333333334
#using a 'rbf' svc model,given the training data, the test resulted with 0.9866666666666667
