# Author: chenqihui
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def warmUpExercise():
    return np.eye(5)


def plotData(x, y):
    fig, ax = plt.subplots()  # create empty figure
    ax.plot(x, y, 'ro', markersize=4)
    ax.set_xlabel("Population of City in 10,000s")
    ax.set_ylabel("Profit in $10,000s")
    return fig


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)  # number of training examples
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha / m) * np.sum((np.dot(X, theta) - y)[:, None] * X, axis=0)
        J_history[i] = computeCost(X, y, theta)
        # print('Cost function: ', J_history[i])

    return (theta, J_history)



def computeCost(X, y, theta):
    m = len(y)
    J = (np.sum((np.dot(X,theta) - y)**2))/(2*m)
    return J


## ==================== Part 1: Basic Function ====================
# print('Running warmUpExercise ... \n')
# print('5x5 Identity Matrix: \n')
# print(warmUpExercise())
# input('Program paused. Press enter to continue.\n');


## ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = pd.read_csv("ex1data1.txt",names=["X","y"])
x = np.array(data.X)[:,None] # population in 10,0000
y = np.array(data.y) # profit for a food truck
m = len(y)
# Plot Data
# fig = plotData(x,y)
# plt.show()
# input('Program paused. Press enter to continue.\n')

## =================== Part 3: Cost and Gradient descent ===================
ones = np.ones_like(x)
X = np.hstack((ones,x))
theta = np.zeros(2)
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')
# compute and display initial cost
J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = ', J)
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
theta = np.array([-1, 2])
J = computeCost(X, y, theta)
print('\nWith theta = [-1 ; 2]\nCost computed = ', J)
print('Expected cost value (approx) 54.24\n')

print('\nRunning Gradient Descent ...\n')
theta, hist = gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent: ')
print(theta[0],"\n", theta[1])
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# plot the linear fit
# plt.plot(x,y,'ro',x,np.dot(X,theta),'b-')
# plt.legend(['Training Data','Linear Regression'])
# plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5],theta)
print('For population = 35,000, we predict a profit of ', predict1*10000)
predict2 = np.dot([1, 7],theta)
print('For population = 70,000, we predict a profit of ', predict2*10000)

## ============= Part 4: Visualizing J(theta_0, theta_1) =============
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i],theta1_vals[j]])
        J_vals[i][j] = computeCost(X,y,t)

# Surface plot
plt.figure()
ax = plt.subplot(111,projection='3d')
Axes3D.plot_surface(ax,theta0_vals,theta1_vals,J_vals,cmap=cm.coolwarm)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.show()
# Contour plot
plt.figure()
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
ax = plt.subplot(111)
C = plt.contour(theta0_vals,theta1_vals,J_vals, np.logspace(-2, 3, 20))
plt.clabel(C,inline=True,fontsize=10)
plt.show()








