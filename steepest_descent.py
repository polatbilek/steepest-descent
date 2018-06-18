# -*- coding: utf-8 -*-
"""

CENG504 OPTIMIZATION METHODS HW2
 ---  Steepest Descent ---

OZAN POLATBILEK
252001021

10.04.2018
"""

import numpy as np
from math import *

phi = (1.0 + sqrt(5.0))/2.0

# searches for best learning_rate
# learning_rate is named as val here
# we want to optimize the f(X - L*d) function here where L is learning rate
def golden_section_searcher(X, d, prev_val, lower, upper, epsilon):
    
    x1 = upper - ((phi - 1)*(upper - lower))
    x2 = lower + ((phi - 1)*(upper - lower))
    val = x1
    
    param2 = X - np.dot(x2, d)
    param2 = param2.tolist()
    
    param1 = X - np.dot(x1, d)
    param1 = param1.tolist()
    
    if equation(param2) < equation(param1):
        if x1 > x2:
            upper = x1
        else:
            lower = x1

    else:
        if x2 > x1:
            upper = x2
        else:
            lower = x2

    if abs(prev_val - val) <= epsilon:
        return val
    else:
        return golden_section_searcher(X, d, val, lower, upper, epsilon)


# derivation idea is similar to limit derivation
# but h doesnt go 0 it is close to zero
# for high dimensions, we take delf
# which has all derivatives
def derivate(f, X):
    h = 0.0000001
    delf = []
    
    for i in range(len(X)):
        E = np.zeros(len(X))
        E[i] = h
        vals = X + E
        delf.append((f(vals) - f(X))/h)
            
    return delf


def difference(X, Y):
    total = 0
    
    for i in range(len(X)):
        total = total + abs(X[i] - Y[i])
    total = total / len(X)
    

    return total


def steepest_descent(X, epsilon):
    
    while True:
        d = derivate(equation, X)
        x_prev = X
        #searching learning rate between [-10,10] is enough
        # learning rate is between 0 and 1 always but lets make it quick and safe one
        learning_rate = golden_section_searcher(X, d, 1, -10, 10, 0.0001)
        X = X - np.dot(learning_rate, d)
        X = X.tolist()
        
        if difference(x_prev, X) < epsilon:
            return x_prev
        
        
    return x_prev


# f(x) = (x-1)^2
# f(x1,x2) = x1^2 + x2^2 - 2x1x2
# f(x1,x2) = x1^2 + 2x2^2 - 5x1
# f(x1,x2,x3) = (x1-2)^2 + (2x2-3)^2 + x3^2
def equation(x):
    #return ((x[0] - 1)**2)
    #return x[0]*x[0] + x[1]*x[1] - (2*x[0]*x[1])
    #return x[0]*x[0]+(2*x[1]*x[1])-5*x[0]
    return (x[0]-2)**2 + (2*x[1]-3)**2 + x[2]**2
    
    

##
## PLEASE GIVE ALL VALUES IN LIST EVEN IF ONE VARIABLE DO IT AS [6]
## ALSO PLEASE MODIFY YOUR EQUATION IN equation(x) FUNCTION. X[0] IS x0, x[1] IS X1
## IF THERE IS ONE VARIABLE IT WILL BE X[0] NOT X
## SOME EQUATION EXAMPLES ARE ABOVE
##
print(steepest_descent([5,6,7], 0.00001))



