import matplotlib.pyplot as plt
import numpy as np

data=np.random.rand(100)
train_data,test_data=np.array(data[:50]),np.array(data[50:])
train_label=np.array(["Class1" if x<=0.5 else "Class2" for x in train_data])

k=[1,2,3,4,5,20,30]
for k in k:
    preds=[]
    
    for test_point in test_data:
        distances=np.abs(train_data-test_point)
        nearest_indices=np.argsort(distances)[:k]
        nearest_labels=[train_label[i] for i in nearest_indices]    
        preds.append(max(set(nearest_labels),key=nearest_labels.count))


    for i in range(50):
         print(f"point {i+51}, value:{test_data[i]:.2f} is {preds[i]}")
    
    plt.scatter(train_data[train_label=="Class1"],[0]*sum(train_label=="Class1"),c="blue",label="Train")
    plt.scatter(train_data[train_label=="Class2"],[0]*sum(train_label=="Class2"),c="red",label="Train")
    preds=np.array(preds)
    plt.scatter(test_data[preds=="Class1"],[1]*sum(preds=="Class1"),c="blue",label="Class 1",marker="x")
    plt.scatter(test_data[preds=="Class2"],[1]*sum(preds=="Class2"),c="red",label="Class 2",marker="x")
    
    plt.title(f"KNN for k={k}")
    plt.legend()
    plt.grid()
    plt.show()
    
