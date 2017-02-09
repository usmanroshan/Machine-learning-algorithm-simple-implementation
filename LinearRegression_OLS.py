# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:21:24 2017

@author: Xueyunzhe
"""

import numpy as np



def ordinary_least_square(x, y):
#    np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    A = x.T.dot(x)
    b = x.T.dot(y)
    return np.linalg.lstsq(A,b)[0]
    
def lr_lasso(x, y, l1_norm):
#    np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y) - 0.5*l1_norm)
    A = x.T.dot(x)
    b = x.T.dot(y) - 0.5*l1_norm
    return np.linalg.lstsq(A,b)[0]

def lr_ridge(x, y, l2_norm):
#    np.linalg.inv(x.T.dot(x) + l2_norm * np.identity(x.shape[1])).dot(x.T).dot(y)
    A = x.T.dot(x) + l2_norm * np.identity(x.shape[1])
    b = x.T.dot(y)
    return np.linalg.lstsq(A,b)[0]

def lr_elasticnet(x, y, l1_norm, l2_norm):
#    np.linalg.inv(x.T.dot(x) + l2_norm * np.identity(x.shape[1])).dot(x.T.dot(y) - 0.5*l1_norm)
    A = x.T.dot(x) + l2_norm * np.identity(x.shape[1])
    b = x.T.dot(y) - 0.5*l1_norm
    return  np.linalg.lstsq(A,b)[0]


class LinearRegression(object):
    '''
    Linear Regression of Ordinary Least Squares
    '''
    def __init__(self, bias = True, lambda_l2 = None, lambda_l1 = None):     
        self.ridge = lambda_l2
        self.lasso = lambda_l1
        self.bias = bias
        self.trained = False
        self.coef_ = None
        
    def fit(self, x, y):        
        if self.bias is True:
            x = np.hstack((x, np.ones(shape=(x.shape[0], 1), dtype=np.float32)))
        
        if (self.ridge is None) and (self.lasso is None):
            self.coef_ = ordinary_least_square(x, y)
        elif (self.ridge is None):
            self.coef_ = lr_lasso(x, y, self.lasso)
        elif (self.lasso is None):
            self.coef_ = lr_ridge(x, y, self.ridge)
        else:
            self.coef_ = lr_elasticnet(x, y, self.lasso, self.ridge)
        
        self.trained = True
        return self
    
    def get_coef(self):
        if self.trained is False:
            print('This model has not been trained')
        elif self.bias is True:
            coef = self.coef_[:-1]
            bias = self.coef_[-1]
            return coef, bias
        else:
            return self.coef_
        
    def predict(self, x):
        if self.trained is False:
            print('This model has not been trained')        
        elif self.bias is True:
            x = np.hstack((x, np.ones(shape=(x.shape[0], 1), dtype=np.float32)))
            return np.dot(x, self.coef_)



#==============================================================================
# Example        
#==============================================================================
 
#Generating train_data and test_data       
x_train = np.random.uniform(-1,1, size = (1000,3))
eta = np.array([0.65, -0.8, 1.3]).reshape(-1,1)
intercept = -0.24
y_train = np.dot(x_train, eta) + intercept    
x_test = np.random.uniform(-1,1, size = (10,3))

#Create instance of LinearRegression                
lr = LinearRegression()
lr.fit(x_train, y_train) #training

coef, bias = lr.get_coef() #Get coefficient and bias
print('coef is:')
print(coef)
print('bias is:')
print(bias)

y_test = lr.predict(x_test) #Predict value
print('Predicted value of LR based on x_test')
print(y_test)
        
        
        
        
        
        
        
        
        
        
        