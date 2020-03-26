
import pandas as pd

dt5 = pd.read_csv("winequality-white.csv")
# print(dt5[1:5])
# print(len(dt5))

# Tap du lieu co 4898 phan tu 12 nhan
# nhan "volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"
dataset8 = dt5[1:9]
dt5.petalLength[1:5]