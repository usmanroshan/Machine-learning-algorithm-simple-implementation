# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:44:18 2017

@author: Xueyunzhe
"""

import numpy as np

def regression(y_, y, batch_size):
    return 0.5 / batch_size * ((y_ - y)**2).sum()


def regression_w_deriv(y_, y, x, batch_size):
    return (1 / batch_size * ((y_ - y) * x).sum(axis = 0)).reshape(-1,1)
    
def regression_b_deriv(y_, y, batch_size):
    return 1 / batch_size * (y_ - y).sum(axis = 0)

def dataiter_index(size, batch_size, shuffle = True):
    index = np.arange(size)
    if shuffle is True:
        np.random.shuffle(index)
    fill_number = batch_size - (size % batch_size)
    if (fill_number != batch_size):
        index = np.hstack((index, index[:fill_number]))
    return index.reshape(-1, batch_size)

def normal_init(coe_shape):
    return np.random.normal(loc=0, scale=1, size = coe_shape).astype(np.float32)
    
def uniform_init(coe_shape):
    return np.random.uniform(low=0, high=1, size = coe_shape).astype(np.float32)

class SGD(object):
    def __init__(self, learning_rate = 0.01):
        self.lr = learning_rate
    
    def update(self, w, wd):
        w = w - self.lr * wd
        return w
    
class LinearRegression(object):
    '''
    Linear Regression of Stochastic Gradient Descent
    '''
    
    def __init__(self, batch_size = None, n_epoch = 10, optimizer = None,
                 init = normal_init, lambda_l1 = 0, lambda_l2 = 0):
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.optimizer = optimizer
        self.initial = init
        self.w = None
        self.b = None
        self.lasso = lambda_l1
        self.ridge = lambda_l2
        self.trained = False
        
    def fit(self, x, y):
        max_n = x.shape[0]
        w = self.initial((x.shape[1],1))
        b = 0.01
        for j in range(self.n_epoch):
            batch_index = dataiter_index(max_n, self.batch_size)
            for i in batch_index:
                y_ = np.dot(x[i], w) + b
                cost = regression(y_, y[i], self.batch_size)# base cost function
                cost = cost + self.ridge * 0.5 / self.batch_size * np.dot(w.T, w) + \
                              self.lasso / self.batch_size * np.linalg.norm(w, 1) # add L1, L2 regulaization
                w_d = regression_w_deriv(y_, y[i], x[i], self.batch_size)
                w_d = w_d + self.ridge * w + self.lasso * np.sign(w)#add weight decay
                w = self.optimizer.update(w, w_d)
                b_d = regression_b_deriv(y_, y[i], self.batch_size)
                b = self.optimizer.update(b,b_d)        
        
        self.w = w
        self.b = b
        self.trained = True
        return self
    
    def get_coef(self):
        if self.trained is False:
            print('This model has not been trained')
        else:
            return self.w, self.b
        
    def predict(self, x):
        if self.trained is False:
            print('This model has not been trained')
        else:
            return np.dot(x, self.w) + self.b
        

#==============================================================================
# Example        
#==============================================================================
 
#Generating train_data and test_data       
x_train = np.random.uniform(-1,1, size = (10000,3))
eta = np.array([0.65, -0.8, 1.3]).reshape(-1,1)
intercept = -0.24
y_train = np.dot(x_train, eta) + intercept    
x_test = np.random.uniform(-1,1, size = (10,3))

#Create instance of LinearRegression and Stochastic Gradient Descent Optimizer
batch_size = 64
n_epoch = 20
learning_rate = 0.01

sgd = SGD(learning_rate = learning_rate) 
lr = LinearRegression(batch_size = batch_size, n_epoch = n_epoch,
                      optimizer = sgd, init = normal_init)

lr.fit(x_train, y_train) #training

coef, bias = lr.get_coef() #Get coefficient and bias
print('coef is:')
print(coef)
print('bias is:')
print(bias)

y_test = lr.predict(x_test) #Predict value
print('Predicted value of LR based on x_test')
print(y_test)
