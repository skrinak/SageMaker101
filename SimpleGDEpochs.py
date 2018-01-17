%matplotlib inline
import math,numpy as np
from numpy.random import random
from matplotlib import pyplot as plt, rcParams, animation, rc
from ipywidgets.widgets import *
rc('animation', html='html5')

#################################################################################
# Define a line as y = mx + b

def lin(m,b,x): return m*x+b

#################################################################################
# Create end values

m=3.
b=8.

#################################################################################
# Create 30 random points. For x set as random. For y use our line definition

n=30
x = random(n)
y = lin(m,b,x)

#################################################################################
x

#################################################################################
y

#################################################################################
# Take a look

plt.scatter(x,y)

#################################################################################
# Define sum of squared errors, loss and average loss functions

def sumSqErr(y,y_pred): return ((y-y_pred)**2).sum()
def loss(y,a,b,x): return sumSqErr(y, lin(m,b,x))
def avg_loss(y,a,b,x): return np.sqrt(loss(y,m,b,x)/n)

#################################################################################
# Set the starting guesses 

m_guess=-1.
b_guess=1.
avg_loss(y, m_guess, b_guess, x)

#################################################################################
# Set a learning rate of 0.01

lr=0.01

#################################################################################
# Update function using derivative of each step
# d[(y-(a*x+b))**2,b] = 2 (b + a x - y)      = 2 (y_pred - y)
# d[(y-(a*x+b))**2,a] = 2 x (b + a x - y)    = x * dy/db

def upd():
    global m_guess, b_guess
    
    # make a prediction using the current weights
    y_pred = lin(m_guess, b_guess, x)
    
    # calculate the derivate of the loss
    dydb = 2 * (y_pred - y)
    dyda = x*dydb
    
    # update our weights by moving in direction of steepest descent
    m_guess -= lr*dyda.mean()
    b_guess -= lr*dydb.mean()
    
#################################################################################
# Run the model

epochs = 10
fig = plt.figure(dpi=100, figsize=(5, 4))
plt.scatter(x,y)
line, = plt.plot(x,lin(m_guess,b_guess,x),'C1')
plt.close()
def animate(i):
    line.set_ydata(lin(m_guess,b_guess,x))
    for i in range(epochs): upd()
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(0, 40), interval=100)
ani
