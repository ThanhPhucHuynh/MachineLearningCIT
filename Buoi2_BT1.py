import pandas as pd 
import numpy as np

from pprint import pprint

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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


from sklearn.datasets import load_iris

iris_dt = load_iris()
iris_dt.data[1:5]
iris_dt.target[1:5]

#a doc du lieu 

dataset1 = pd.read_csv('winequality-white.csv',
                        names = ["facidity", "vacidity", "citric", "sugar", "chlorides", "fsulfur", 
                            "tsulfur", "density", "pH", "sulphates", "alcohol", "quality"],
                            sep=";",skiprows=1)


data = dataset1

haha = ID3(data[0:100],data[0:100],data[0:100].columns[:-1],'quality',None)


nhan=np.unique(data.quality)
print(bcolors.OKBLUE+"B-C: tap du lieu co", len(data), "phan tu 7 loai nhan" , np.unique(data.quality))
# tap du lieu co 4899 phan tu 7 loai nhan [3 4 5 6 7 8 9]
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(data.iloc[:,1:11],data.quality,test_size=0.2, random_state=100)



from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion="gini",random_state=100, max_depth=11,min_samples_leaf=5)
# print(clf_gini)
clf_gini.fit(X_train,y_train)


y_pred = clf_gini.predict(X_test)
# y_test = clf_gini.predict([[4,4,3,3]])
# print(y_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print(bcolors.OKGREEN+"D:danh gia do chinh xac tong the va tung lop: \n"+bcolors.ENDC)
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
np.unique(data)

a =confusion_matrix(y_test,y_pred,labels= nhan)
print(a)

nhan=np.unique(data.quality)

X_train, X_test,y_train,y_test = train_test_split(data.iloc[:,1:11],data.quality,test_size=0.2, random_state=100)

clf_gini = DecisionTreeClassifier(criterion="entropy",random_state=100, max_depth=11,min_samples_leaf=5)

clf_gini.fit(X_train,y_train)

y_pred = clf_gini.predict(X_test)
print(bcolors.OKGREEN+"D:danh gia do chinh xac tong the va tung lop cho 6 ptu dau: \n"+bcolors.ENDC)

print("Accuracy is ", accuracy_score(y_test[0:6],y_pred[0:6])*100)
np.unique(data)
a =confusion_matrix(y_test[0:6],y_pred[0:6],labels= nhan)
print(a)
print(bcolors.OKBLUE+"Cay quyet dinh cho data[0:100] \n"+bcolors.ENDC)
print(haha)
# print(bcolors.WARNING + "Warning: No active frommets remain. Continue?" + bcolors.ENDC+"ádasd")