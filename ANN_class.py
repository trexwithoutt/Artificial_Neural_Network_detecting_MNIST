import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class ANN():
    def __init__(self, nhid, nunits):
        self.nhid = nhid
        self.nunits = nunits # a list of number of units for each layer
        self.weights = []
        self.bias =  []
        self.result = [None]* (self.nhid+2)
        self.delta =  [None]* (self.nhid+2)
        self.zs = [None] * (self.nhid+2)
        self.errors = []
        self.accuracies = []
        
    
    def forward(self, x):
        '''
        The forward method is basically going through layer by layer, 
        and have sigmoid as activation function to generate the output
        for each layer
        
        Input:
            x: an input row vector
        Output:
            last layer result
        '''
        self.result[0] = np.array(x)
        for layer in range(1, self.nhid+2):
            z = self.result[layer-1]@self.weights[layer-1] + self.bias[layer-1]
            self.zs[layer] = np.array(z)
            result = self.sigmoid(z)
            self.result[layer] = np.array(result)
        return self.result[layer]
    
    def backward(self, x, y):
        '''
        The backward method is doing the back probagation method for each layer.
        To ease the weights updating, I saved the delta for each layer
        
        Input: 
            x: input row vector
            y: true result
        '''
        self.delta[self.nhid+1] = self.dsigmoid(self.zs[self.nhid+1]) * (y - self.result[self.nhid+1])
        for layer in range(self.nhid, 0, -1):
            z = self.zs[layer]
            self.delta[layer] = (self.delta[layer+1]@self.weights[layer].T) * self.dsigmoid(z)
                
                
    def weights_update(self, alpha):
        '''
        This function basically update the parameters(not only weights) for each layer
        based on the delta saved from the back probagation method
        
        Input:
            alpha: learning rate
        '''
        for layer in range(self.nhid+1):
            self.weights[layer] += alpha * self.result[layer].T@self.delta[layer+1]
            self.bias[layer] += alpha * self.delta[layer+1]
    
    def fit(self, X, y, alpha, t):
        '''
        Fitting the neural network and calculate errors and accuracy for each epoch
        
        Input:
            X: training features
            y: training result 
            alpha: learning rate
            t: number of iterations(epoch)
        
        '''
        y = np.asarray(pd.get_dummies(y), dtype=float)
        m, n = X.shape
        self.param_init(X,y)
        for epoch in range(t):
            error =0
            for i in range(m):
                x = np.asarray(X[i,:])
                v = np.asarray(y[i,:]).reshape((1, len(y[i,:])))
                
                a = self.forward(x)
                self.backward(x, v)
                self.weights_update(alpha)
                error += np.sum((v-a)**2)
            accuracy = self.accuracy(X, y)  
            self.accuracies.append(accuracy)
            self.errors.append(error/len(X))
            if epoch%5 ==0:
                print('>epoch=%d,error=%.3f'%(epoch, error/len(X)))
                
            
    def predict(self, T):
        '''
        Input:
            T: test examples
            
        Output:
            predict result
        '''
        result = T
        for layer in range(1, self.nhid+2):
            z = result@self.weights[layer-1]
            result = self.sigmoid(z)
        return np.argmax(result, axis=1)
        
    def print(self):
        print(f'current weights: \n {self.weights}' \
             f'current bias: \n {self.bias}')
    
    def param_init(self, X, y):
        '''
        Initialing parameters for neural network by using xavier initialization
        
        Input:
            X: training features
            y: training result
        '''
        nin, nout = X.shape[1], self.nunits[0]
        self.weights.append(self.xavier_init(nin, nout))
        self.bias.append(self.xavier_init(1, nout))
        for i in range(self.nhid - 1):
            self.weights.append(self.xavier_init(self.nunits[i], self.nunits[i+1]))
            self.bias.append(self.xavier_init(1, self.nunits[i+1]))
        self.weights.append(self.xavier_init(self.nunits[-1], y.shape[1]))
        self.bias.append(self.xavier_init(1, y.shape[1]))
        
    def xavier_init(self, nin, nout):
        '''
        Input:
            nin: number of input
            nout: number of output
        Output:
            xavier initialization
        '''
        up = np.sqrt(6 / (nin + nout))
        low = -up
        return (up - low) * np.random.random_sample((nin, nout)) + low
    
    def accuracy(self, X, y):
        '''
        Input:
            X: training features
            y: training result
        Output:
            accuracy
            
        '''
        predictions = self.predict(X).T.ravel()
        predictions = np.squeeze(np.asarray(predictions))
        label = np.argmax(y, axis=1)
        return sum(predictions == label) / X.shape[0]
                
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def dsigmoid(self, x):
        '''
        derivatives for sigmoid
        '''
        return self.sigmoid(x) * (1 - self.sigmoid(x))