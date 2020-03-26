import pandas as pd 
import numpy as np
from pprint import pprint

# print("aaa")

dataset = pd.read_csv('play_tennis.csv')
# ,
                        # names=['day','outlook','temp','humidity','wind','play'])
# print(dataset)
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








dataset2 = dataset.drop('day',1);


haha = ID3(dataset2,dataset2,dataset2.columns[:-1],'play',None)
print(dataset2.columns[:-1])
print(dataset2)
# print('aaa\n')
# print(dataset.drop('day',1))
# print(dataset,dataset.columns[:-1])
print(haha)




