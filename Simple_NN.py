# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:56:10 2018

@author: sgnka
"""


#Package import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils_file import plot_decision_boundary, sigmoid, load_planar_dataset
np.random.seed(1)

#Load Dataset
X, Y = load_planar_dataset()

# Visualize the data:
sns.scatterplot(x=X[0, :], y=X[1, :],hue=Y[0],palette='dark')

def data_shape(X,Y):
    #No of data and training examples
    shape_X = X.shape
    shape_Y = Y.shape
    m = Y.shape[1]  # training set size
    
    print ('The shape of X is: ' + str(shape_X))
    print ('The shape of Y is: ' + str(shape_Y))
    print ('I have m = %d training examples!' % (m))

#Test on Regression Model
from sklearn_regression import pl
pl(X,Y)

#Defining Neural Network Size
def layer_sizes(X,Y):
    
    n_x = X.shape[0]     #n_x -- size of the input layer
    n_h = 4             # n_h -- size of the hidden layer
    n_y = Y.shape[0]    #n_y -- size of the output layer
    return (n_x,n_h,n_y)

#Initializing the model's parameterh
def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(2) 
    W1= np.random.randn(n_h,n_x) *0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


#Forward propagation   
def forward_propagation(X,parameters):
    #Retrieving values
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    #Implement Forward Propagation to calculate A2
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache


#Cost function
def compute_cost(A2,Y,parameters):
     m = Y.shape[1] # number of example
     W1 = parameters['W1']
     W2 = parameters['W2']
     # Compute the cross-entropy cost 
     logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
     cost = -np.sum(logprobs) / m
    
     cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
     assert(isinstance(cost, float))
     return cost
 
    
#Backward Propagation
def backward_propagation(parameters,cache,X,Y):
    m=X.shape[1]
    #Retrive W1,W2,A1,A2
    W1 = parameters['W1']
    W2 = parameters['W2'] 
    A1 = cache['A1']
    A2 = cache['A2']
    
    #Backprop start
    dZ2 = A2-Y
    dW2 = 1/m* np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

def update_parameters(parameters, grads, learning_rate=1.2):
    #Retrive W1,W2,b1,b2,dw1,dw2,db1,db2
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    #Update
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    #save updated to parameters dict
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


#Final Function
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
   
    parameters = initialize_parameters(n_x, n_h, n_y) 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
     # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)
        
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
    
    return parameters

def predict(parameters, X):
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X,parameters)
    predictions = np.round(A2)
    
    
    return predictions

# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')