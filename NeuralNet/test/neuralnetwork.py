

import scipy.io
import matplotlib.pyplot as plt
import numpy as np



# load the pictures
mat = scipy.io.loadmat("ex4data1.mat")
X = mat["X"]
Y1 = mat["y"]
# replace 10 with 0 because the data is prepared for octave and there is no 0, 10 was used instead
Y1 = np.where(Y1==10, 0, Y1) 
Y = []
for i in range(len(Y1)):
    arr = np.zeros(10)
    arr[Y1[i][0]] = 1
    Y.append(arr)

Y = np.array(Y)

img = np.reshape(X[0], (-1, 2))
# img = X[0].resize(20,20)
plt.imshow(img)

print(type(X))