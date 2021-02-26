import numpy as np
from sklearn.model_selection import train_test_split
data = np.load("./data.npy")
label = np.load("./label.npy")
# print(data)
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=10)
