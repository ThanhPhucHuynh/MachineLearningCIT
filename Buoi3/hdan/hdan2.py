from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target
# print(x,y)

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size= .3 , random_state=0)



model = GaussianNB()
model.fit(X_train, y_train)
print(model)

thucte = y_test
dubao = model.predict(X_test)

print(thucte)
print(dubao)

from sklearn.metrics import confusion_matrix
cnf_matrix_gnb = confusion_matrix(thucte,dubao)
print(cnf_matrix_gnb)

from sklearn.model_selection import KFold
kf = KFold(n_splits=15)
print(type(x))
print(y)

for train_index, test_index in kf.split(x):
    print("TRAIN:", train_index, "TEST:",test_index )
    X_train,X_test = x[train_index,],x[test_index,]
    y_train,y_test = y[train_index,],y[test_index,]
    print("X_test:",X_test)
    print("===============")


