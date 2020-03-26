import pandas as pd 
import numpy as np

from pprint import pprint

# print("aaa")

dataset = pd.read_csv('winequality-white.csv',
                        names = ["facidity", "vacidity", "citric", "sugar", "chlorides", "fsulfur", 
                            "tsulfur", "density", "pH", "sulphates", "alcohol", "quality"],
                            sep=";",skiprows=1)
def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts=True)
    en=0
    for i in range(len(elements)):
        pi = counts[i]/np.sum(counts)
        temp = - pi*np.log2(pi)
        en = en + temp
    return en

def InfoGain(data,split_attribute_name,target_name):

    total_entropy = entropy(data[target_name])

    vals,counts = np.unique(data[split_attribute_name], return_counts=True)
    print(vals)
    print(counts)
    total_elements = np.sum(counts)
    print(total_elements)
    Weighted_Entropy=0
    for i in range(len(vals)):
        print(vals[i])
        print(split_attribute_name)

        Weighted_Elements = (counts[i]/total_elements)
        print(Weighted_Elements)

        dt_split_attibute_vals = data[data[split_attribute_name] == vals[i]]
        print(dt_split_attibute_vals)

        Entropy_Elements = entropy(dt_split_attibute_vals[target_name])

        Weighted_Entropy = Weighted_Entropy +Weighted_Elements*Entropy_Elements
        print(Weighted_Entropy)
    Information_Gain = total_entropy - Weighted_Entropy
    print(Information_Gain)
    return Information_Gain
# InfoGain(dataset,"outlook","play")

def ID3(data,originaldata,features,target_attribute_name,parent_node_class):

    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])\
            [np.argmax(np.unique(data[target_attribute_name],\
                                    return_counts=True)[1])]
        print('parent_note_class:',parent_node_class)

        item_values = [InfoGain(data,feature,target_attribute_name)\
                    for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature:{}}
        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class)
            tree[best_feature][value] = subtree
        return(tree)



from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


X=[[180,15,0],
    [167,42,1],
    [136,35,1],
    [174,15,0],
    [141,28,1]
]
Y=['nam','nu','nu','nam','nu']
# print(type(data))
data1 = np.array(X)
# print(a)
ad= pd.DataFrame(data=data1,columns=['chieucao', 'do daitoc', 'giongnoi'])
ad['nhan'] =pd.Series(data=np.array(Y),index=ad.index)
haha = ID3(ad,ad,ad.columns[:-1],'nhan',None)
print(haha)
X_train, X_test,y_train,y_test = train_test_split(ad.iloc[:,0:3],ad.nhan,test_size=1/3.0, random_state=5)
clf_gini = DecisionTreeClassifier(criterion="gini",random_state=5, max_depth=3,min_samples_leaf=5)
# print(clf_gini)
clf_gini.fit(X_train,y_train)

y_pred = clf_gini.predict(X_test)
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
# np.unique(data)

a =confusion_matrix(y_test,y_pred)
print(a)


X=[[180,15,0],
    [167,42,1],
    [136,35,1],
    [174,15,0],
    [141,28,1],
    [135,39,1]
]
Y=['nam','nu','nu','nam','nu','nam']
# print(type(data))
data1 = np.array(X)
# print(a)
ad= pd.DataFrame(data=data1,columns=['chieucao', 'do daitoc', 'giongnoi'])
ad['nhan'] =pd.Series(data=np.array(Y),index=ad.index)
# haha = ID3(ad,ad,ad.columns[:-1],'nhan',None)
# print(haha)
X_train, X_test,y_train,y_test = train_test_split(ad.iloc[:,0:3],ad.nhan,test_size=5/6, random_state=5)
clf_gini = DecisionTreeClassifier(criterion="gini",random_state=5, max_depth=3,min_samples_leaf=5)
# print(clf_gini)
clf_gini.fit(X_train,y_train)

y_pred = clf_gini.predict(X_test)
# y_test = clf_gini.predict([[135,39,1,'nam']])
# print(y_test)
# print(y_train)
# print(X_test)
# print(X_train)
# from sklearn.metrics import accuracy_score, confusion_matrix
# print("D:danh gia do chinh xac tong the va tung lop: \n")
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
# np.unique(data)

a =confusion_matrix(y_test,y_pred)
print(a)
print("ti le nam cua nguoi  [135,39,1] nay la 40% vay xac xuat 60% cho nguoi nay la nữ")