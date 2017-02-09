# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 17:53:41 2017

@author: Xueyunzhe
"""
import numpy as np
import random
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1-x)

def normal_init(coe_shape):
    return np.random.normal(loc=0, scale=0.1, size = coe_shape).astype(np.float32)
    
def uniform_init(coe_shape):
    return np.random.uniform(low=0, high=1, size = coe_shape).astype(np.float32)

class SGD(object):
    def __init__(self, learning_rate = 0.01):
        self.lr = learning_rate
    
    def update(self, w, wd):
        w = w - self.lr * wd
        return w
    
class NeuralNetwork(object):
    '''
    Multilayer Perceptron / Fully connected Network
    
    batch_size is 1
    '''
    
    def __init__(self, layers, activation = 'sigmoid', init = normal_init):
        self.layers = layers
        self.optimizer = None
        self.trained = False

        if activation == 'sigmoid':
            self.act = sigmoid
            self.act_deriv = sigmoid_deriv
        
        self.w = [normal_init((layers[i], layers[i+1])) for i in range(len(layers) - 1)]
        self.b = [np.ones((layers[i+1])) for i in range(len(layers) - 1)]
        
    def forward(self, x):
        self.a = [] #activation values list
        self.a.append(x) # set input as the first activation values
        for i in range(len(self.layers) - 1):
            self.a.append(self.act(np.dot(self.a[i], self.w[i]) + self.b[i]))

    
    def backpropagation(self, y):
        self.err = []
        self.b_d = []
        self.w_d = []
       
        self.err.append((self.a[-1] - y) * self.act_deriv(self.a[-1]))
        for i in range(len(self.layers) - 2):
            self.err.append(
            (self.err[i] * self.w[-1-i]).sum(axis=1) * self.act_deriv(self.a[-2-i])
            )
        self.err.reverse()
        for i in range(len(self.w)):
            self.w_d.append(self.err[i] * self.a[i].reshape(-1,1))
        for i in range(len(self.w)):
            self.w[i] = self.optimizer.update(self.w[i], self.w_d[i])
        
        for i in range(len(self.b)):
            self.b_d.append(self.err[i])
        for i in range(len(self.b)):
            self.b[i] = self.optimizer.update(self.b[i], self.b_d[i])
            
    def fit(self, x, y, n_epoch = None, optimizer = None):
        self.n_epoch = n_epoch
        self.optimizer = optimizer
        
        for i in range(self.n_epoch):
            for m,n in zip(x,y):
                self.forward(m)
                self.backpropagation(n)
       
        self.trained = True        
        return self
                
    def predict(self, x, prob=False):
        if self.trained is False:
            print('This model has not been trained')
        else:        
            self.forward(x)
            y_pred = self.a[-1]
            if prob is True:
                return y_pred
            else:
                yp = np.zeros_like(y_pred)
                yp[y_pred>0.5] = 1
                return yp

    def score(self,y1,y2):
        accuracy =1- abs(y1-y2).mean()
        print (accuracy)             


m = np.random.randn(400,3)+5
n = np.random.randn(400,3)+[8,2,0]
t = np.random.randn(200,3)+[2,8,0]
n = np.vstack((n,t))
m[:,2]=1
n[:,2]=0


sca = StandardScaler()

temp = np.vstack((m,n)).tolist()
random.shuffle(temp)
data = np.array(temp)
x = data[:,:2]
y = data[:,2]
x=sca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
batch_size = 64
n_epoch = 100
learning_rate = 0.1

sgd = SGD(learning_rate = learning_rate) 

NN = NeuralNetwork(layers=[2,2,1])
NN.fit(x_train, y_train, n_epoch = n_epoch, optimizer=sgd)
yp = NN.predict(x_test)
NN.score(yp,y_test.reshape(-1,1))                
            