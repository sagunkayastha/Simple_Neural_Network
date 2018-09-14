
"""
Created on Fri Sep 14 12:19:29 2018

@author: sgnka
"""

# Functions
## 1) layer_sizes(X,Y)
 Arguments:
 
 X -- input dataset of shape (input size, number of examples
 
 Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    
## 2) initialize_parameters(n_x, n_h, n_y)
 Argument:
 
    n_x -- size of the input layer
    
    n_h -- size of the hidden layer
    
    n_y -- size of the output layer
    
    Returns:
    
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)

					
## 3) forward_propagation(X, parameters):
 
    Argument:
    
    X -- input data of size (n_x, m)
    
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    
    A2 -- The sigmoid output of the second activation
    
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
	
J=−1/m ∑(y(i)log(a[2](i))+(1−y(i))log(1−a[2](i)))	
## 4) compute_cost(A2, Y, parameters):
   
   Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    
    Y -- "true" labels vector of shape (1, number of examples)
    
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    
    cost -- cross-entropy cost given equation (13)
    
## 5) backward_propagation(parameters, cache, X, Y):
    
    Implement the backward propagation using the instructions above.
    
    Arguments:
    
    parameters -- python dictionary containing our parameters 
    
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    
    X -- input data of shape (2, number of examples)
    
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    
    grads -- python dictionary containing your gradients with respect to different parameters
	
	{
	dZ2 = A2-Y
	
	dW2 = 1/m * dZ1 . A1.T
	
	db1 = 1/m * sum( dZ2)
	
	dZ1 = W2.T . dZ2 *' g'(Z1) # *'= element wise multiplication
	}
	
	
## 6) update_parameters(parameters, grads, learning_rate):
    
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    
    parameters -- python dictionary containing your parameters 
    
    grads -- python dictionary containing your gradients 
    
    Returns:
    
    parameters -- python dictionary containing your updated parameters 
    
	
## 7) def nn_model(X, Y, n_h, num_iterations, print_cost):
### Final Model
    Arguments:
    
    X -- dataset of shape (2, number of examples)
    
    Y -- labels of shape (1, number of examples)
    
    n_h -- size of the hidden layer
    
    num_iterations -- Number of iterations in gradient descent loop
    
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    
    parameters -- parameters learnt by the model. They can then be used to predict.
    
  
## 8) predict(parameters, X):
    
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    
    parameters -- python dictionary containing your parameters 
    
    X -- input data of size (n_x, m)
    
    Returns
    
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
  
