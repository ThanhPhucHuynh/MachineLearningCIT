import pandas as pd
import numpy as np



wineWhite = pd.read_csv("winequality-white.csv",
 names = ["facidity", "vacidity", "citric", "sugar", "chlorides", "fsulfur", 
                            "tsulfur", "density", "pH", "sulphates", "alcohol", "quality"], 
                            sep=";",quotechar='"',skiprows=1)
# print(wineWhite.columns)
# print(wineWhite['quality'].unique())
# nhan=np.unique(wineWhite)
# print(nhan)

# print(len(wineWhite['quality'].unique()))
# print(len(wineWhite.columns))

# print(wineWhite.iloc[:,1:11])
# data.iloc[:,1:11]
#2

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wineWhite.iloc[:,1:11],wineWhite.quality,test_size= .3, random_state=0)
x=wineWhite.iloc[:,1:11].to_numpy()
y=wineWhite.quality.to_numpy()









# model = GaussianNB()
# model.fit(X_train, y_train)
# print(model)

# thucte = y_test
# dubao = model.predict(X_test)
# from sklearn.metrics import accuracy_score
# print(thucte)
# print(dubao)
# print('accuracy = ',accuracy_score(y_test, dubao))
# from sklearn.metrics import confusion_matrix
# cnf_matrix_gnb = confusion_matrix(thucte,dubao)
# print(cnf_matrix_gnb)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
kf = KFold(n_splits=70,shuffle=False)
i=0
for train_index, test_index in kf.split(x):
    print("TRAIN:", train_index, "TEST:",test_index )
   
    X_train,X_test = x[train_index,],x[test_index,]
    y_train,y_test = y[train_index,],y[test_index,]
    # print("X_test:",X_test)


            
    model = GaussianNB()
    model.fit(X_train, y_train)
    # print(model)

    thucte = y_test
    dubao = model.predict(X_test)
    
    # print(thucte)
    # print(dubao)
    print('accuracy = ',accuracy_score(y_test, dubao))
    i+=accuracy_score(y_test, dubao)
    # cnf_matrix_gnb = confusion_matrix(thucte,dubao)
    # print(cnf_matrix_gnb)



    print("===============")
# print(x,y,i)



# from sklearn.tree import DecisionTreeClassifier
# clf_gini = DecisionTreeClassifier(criterion="gini",random_state=100, max_depth=11,min_samples_leaf=5)
# # print(clf_gini)
# clf_gini.fit(X_train,y_train)
# y_pred = clf_gini.predict(X_test)
# print('accuracy = ',accuracy_score(y_test, y_pred))
print(float(i)/70)