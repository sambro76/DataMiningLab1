import numpy as np
from sklearn.model_selection import cross_val_score

ATT=np.genfromtxt('ATNTFaceImages400.txt', delimiter=',')
split_cols=1
rows = len(ATT) 
cols=ATT.shape[1]

train=np.zeros((rows, (cols - split_cols*40)))
test=np.zeros((rows, split_cols*40))
#Splitting the dataset into Train & Test Data

  
X_train=train[1:,:].transpose()
X_test=test[1:,:].transpose()
y_train=train[0,:]
y_test=test[0,:]

print("\n#######kNN (k=5)#######")
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("kNN Test set predictions: ", knn.predict(X_test))
print("kNN Test set accuracy: ", knn.score(X_test, y_test))
#cross_validation
scores = cross_val_score(knn, X_train, y_train, cv=5)
print("kNN Cross-Val scores: ", scores)
print("kNN Average Cross-Val score: ", scores.mean())

print("\n#######Nearest Centroid#######")
from sklearn.neighbors.nearest_centroid import NearestCentroid
nc=NearestCentroid()
nc.fit(X_train, y_train)
print("Centroid Test set predictions: ", nc.predict(X_test))
print("Centroid Test set accuracy: ", nc.score(X_test, y_test))
#cross_validation
scores = cross_val_score(nc, X_train, y_train, cv=5)
print("Centroid Cross-Val scores: ", scores)
print("Centroid Average Cross-Val score: ", scores.mean())

print("\n#######Logistic Regression#######")
from sklearn import linear_model
lr=linear_model.LogisticRegression()
lr.fit(X_train, y_train)
print("Logistic Regression Test set predictions: ", lr.predict(X_test))
print("Logistic Regression set accuracy: ", lr.score(X_test, y_test))
#cross_validation
scores = cross_val_score(lr, X_train, y_train, cv=5)
print("Logistic Regression CV scores: ", scores)
print("Logistic Regression Average CV score: ", scores.mean())

print("\n#######SVM#######")
from sklearn import svm
svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(X_train, y_train) 
print("SVM Test set predictions: ", svm_clf.predict(X_test))
print("SVM set accuracy: ", svm_clf.score(X_test, y_test))
#cross_validation
scores = cross_val_score(svm_clf, X_train, y_train, cv=5)
print("SVM Cross-Val scores: ", scores)
print("SVM Average Cross-Val score: ", scores.mean())
