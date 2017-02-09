# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:19:08 2017

@author: Xueyunzhe
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logloss(y_, y, batch_size):
    return -1 / batch_size * (y * np.log(y_) + (1-y) * np.log(1 - y_)).sum()

def logloss_w_deriv(y_, y, x, batch_size):
    return (1 / batch_size * ((y_ - y) * x).sum(axis = 0)).reshape(-1,1)
    
def logloss_b_deriv(y_, y, batch_size):
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
    
class LogisticRegression(object):
    '''
    Logistic Regression of Stochastic Gradient Descent
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
        b = 0.00
        for j in range(self.n_epoch):
            batch_index = dataiter_index(max_n, self.batch_size)
            for i in batch_index:
                a = np.dot(x[i], w) + b
                y_ = sigmoid(a)
                cost = logloss(y_, y[i], self.batch_size)# base cost function
                cost = cost + self.ridge * 0.5 / self.batch_size * np.dot(w.T, w) + \
                              self.lasso / self.batch_size * np.linalg.norm(w, 1) # add L1, L2 regulaization
                w_d = logloss_w_deriv(y_, y[i], x[i], self.batch_size)
                w_d = w_d + self.ridge * w + self.lasso * np.sign(w)#add weight decay
                w = self.optimizer.update(w, w_d)
                b_d = logloss_b_deriv(y_, y[i], self.batch_size)
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
            a = np.dot(x, self.w) + self.b
            y = sigmoid(a)
            return y

#==============================================================================
# Example        
#==============================================================================
 
#Generating train_data and test_data       
x_train = np.random.normal(5,2, size = (10000,2))
y_train = np.ones((10000,1))
y_train[x_train[:,0] > x_train[:,1]] = 0  
x_test = x = np.array([[3,1],[2,4]])

#Create instance of LinearRegression and Stochastic Gradient Descent Optimizer
batch_size = 64
n_epoch = 20
learning_rate = 0.01

sgd = SGD(learning_rate = learning_rate) 
lr = LogisticRegression(batch_size = batch_size, n_epoch = n_epoch,
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
