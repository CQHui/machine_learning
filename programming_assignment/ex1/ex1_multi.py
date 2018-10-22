# Author: chenqihui
import numpy as np
import pandas as pd


def normalEqn(X, y):
    return np.dot((np.linalg.inv(np.dot(X.T, X))), np.dot(X.T, y))

def computeCost(X, y, theta):
    m = len(y)
    J = (np.sum((np.dot(X,theta) - y)**2))/(2*m)
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y)  # number of training examples
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha / m) * np.sum((np.dot(X, theta) - y)[:, None] * X, axis=0)
        J_history[i] = computeCost(X, y, theta)
        # print('Cost function: ', J_history[i])

    return (theta, J_history)

# ================ Part 1: Feature Normalization ================
print('Plotting Data ...\n')
data = pd.read_csv("ex1data2.txt",names=["sz","bed","price"])
s = np.array(data.sz)
r = np.array(data.bed)
p = np.array(data.price)
m = len(r)
s = np.vstack(s)
r = np.vstack(r)
X = np.hstack((s,r))
y = np.vstack(p)
print('First 10 examples from the dataset: \n')
print(' x = %s ,\n y = %s \n' % (X[:10], y[:10]))
X = np.hstack((np.ones_like(s),X))

theta = normalEqn(X, p)

print('Theta computed from the normal equations: \n')
print(theta)
print('\n')


# Estimate the price of a 1650 sq-ft, 3 br house
price = np.dot([1,1650,3],theta)


print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): \n',
       price)