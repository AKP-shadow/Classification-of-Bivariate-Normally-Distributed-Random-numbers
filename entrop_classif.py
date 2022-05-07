
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# read data
x1 = np.random.multivariate_normal([30,30], [[3, 0.5], [0.5, 4]], 100)
y1= np.random.multivariate_normal([34.5,34.5], [[4,1],[1,3]], 100)

#sessioning the random data
with open("set1.csv","r+") as set1:
    writer = csv.writer(set1)
    writer.writerows(x1)
with open("set2.csv","r+") as set2:
    writer = csv.writer(set2)
    writer.writerows(y1)

X= np.concatenate((x1,y1),axis =0 )

y= np.array([0]*100 + [1]*100)

log_reg = LogisticRegression()
log_reg.fit(X, y)


parameters = log_reg.coef_[0]

parameter0 = log_reg.intercept_

# Plotting the decision boundary
fig = plt.figure()
x_values = [np.min(X[:, 1] -5 ), np.max(X[:, 1] +5 )]

y_values = np.dot((-1./parameters[1]), (np.dot(parameters[0],x_values) + parameter0))
colors=['green' if l==0 else 'red' for l in y]
plt.scatter(X[:, 0], X[:, 1], label='Logistics regression', color=colors)
plt.plot(x_values, y_values, label='Decision Boundary')
plt.title("Classification of two sets of 2D Normal distribution")

plt.show()