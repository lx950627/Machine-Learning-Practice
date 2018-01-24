import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
'''
mean=[0,0]
cov=[[1,0],[0,1]]
x, y = np.random.multivariate_normal(mean, cov, 1000).T
plt.scatter(x, y)
plt.title("Figure a")
plt.show()

mean=[1,1]
x, y = np.random.multivariate_normal(mean, cov, 1000).T
plt.scatter(x, y)
plt.title("Figure b")
plt.show()

mean=[0,0]
cov=[[2,0],[0,2]]
x, y = np.random.multivariate_normal(mean, cov, 1000).T
plt.scatter(x, y)
plt.title("Figure c")
plt.show()

cov=[[1,0.5],[0.5,1]]
x, y = np.random.multivariate_normal(mean, cov, 1000).T
plt.scatter(x, y)
plt.title("Figure d")
plt.show()

cov=[[1,-0.5],[-0.5,1]]
x, y = np.random.multivariate_normal(mean, cov, 1000).T
plt.scatter(x, y)
plt.title("Figure e")
plt.show()
'''
m=[[1,0],[1,3]]
w, v = LA.eig(m)
print(v[np.argmax(w)])