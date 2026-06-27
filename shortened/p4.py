import pandas as pd

data=pd.read_csv("../data.csv")
attributes=data.columns[:-1]
class_label=data.columns[-1]
hypothesis=['0' for _ in attributes]


for index,row in data.iterrows():
    if row[class_label]=='yes':
        for i,value in enumerate(row[attributes]):
            if hypothesis[i]=='0' or hypothesis[i]==value:
                hypothesis[i]=value
            else:
                hypothesis[i]='?'

print(hypothesis)