import numpy as np
from math import exp
from scipy.stats import logistic
import matplotlib.pyplot as plt
import pandas as pd

class Layer:
    '''
    The class Layer contains the parameters of each layer. Its initialization make them all empty
    '''
    def __init__(self):
        self.W=[] # self.W = the incoming weights
        self.b=[] # self.b = the biases
        self.a=[] # self.a = the activations
        self.z=[] # self.z = the outputs
        self.d_W=[] # self.d_W = the gradient of the incoming weights
        self.d_b=[] # self.d_b = the gradient of the biases
        self.d_a=[] # self.d_a = the gradient of the activations
        self.d_z=[] # self.d_z = the gradient of the outputs

class MLP(Layer): # Multi Layer Perceptron
    def __init__(self, neurons_per_layer):
        '''
        Create the weight matrices for each layer following the neurons_per_layer vector.
        It initializes also the loss and accuracy vector
        
        self.layer[0].W contains the weights which connect input layer 1 with 1st hidden layer. Dimensions [n_1st,n_input]
        self.layer[0].b contains the biases of 1st hidden layer
        self.layer[0].a contains the activation of 1st hidden layer
        self.layer[0].z contains the outputs of 1st hidden layer
        self.layer[0].d_W contains the derivative of loss w.r.t the weights which connect input layer 1 with 1st hidden layer. Dimensions  [n_1st,n_input]
        self.layer[0].d_b contains the derivative of loss w.r.t the biases of 1st hidden layer
        self.layer[0].d_a contains the derivative of loss w.r.t the activations of 1st hidden layer
        self.layer[0].d_z contains the derivative of loss w.r.t the outputs of 1st hidden layer
        self.layer[1].W contains the weights which connect 1st hidden layer with 2nd hidden layer. Dimensions [n_2nd,n_1st]
        self.layer[1].b contains the biases of 2nd hidden layer
        ecc...
        self.weights[n] contains the weights which connect nth hidden layer with output layer. Dimensions Dimensions [n_nth,n_output]
        self.biases[n] contains the biases of output layer
        ...
        
        INPUT : 
        - neurons_per_layer : numpy array containing the number of neurons in
            [ input layer, hidden layer1, hidden layer 2, ..., output layer ]
        '''
        super().__init__()
        
        self.layer={}
        
        for i in range(0,len(neurons_per_layer)-1) :
            self.layer[i]=Layer()
            self.layer[i].W=(10**(-1))*np.random.randn(neurons_per_layer[i+1],neurons_per_layer[i])
            self.layer[i].b=np.zeros((1,neurons_per_layer[i+1]))
            self.layer[i].a=np.zeros((1,neurons_per_layer[i+1])) 
            self.layer[i].z=np.zeros((1,neurons_per_layer[i+1]))
            self.layer[i].d_W=np.zeros((neurons_per_layer[i+1],neurons_per_layer[i]))
            self.layer[i].d_b=np.zeros((1,neurons_per_layer[i+1])) 
            self.layer[i].d_a=np.zeros((1,neurons_per_layer[i+1]))
            self.layer[i].d_z=np.zeros((1,neurons_per_layer[i+1]))

        
        self.losses=[]
        self.accuracies=[]
        
    def sigmoid(a) :
        '''
        Sigmoid activation function. It can work with single inputs or vectors or matrices.
        '''
        # logistic.cdf from scipy is used for stability instead of exponential functions
        
        return np.array(logistic.cdf(a)) 
    
    def d_sigmoid(a) :
        '''
        Derivative of sigmoid activation function. It can work with single inputs or vectors or matrices.
        Return the sigmoid derivative of a
        '''
        s = MLP.sigmoid(a)
        res = s*(1-s)
        return(res)
    
    def forward(self, x) :
        '''
        Forward function. From input layer to output layer. Input can handle 1D or 2D inputs.

        INPUTS:
        - x : numpy array of size NxD, where N is the number of samples, D is the number of input dimensions referred as n_input before

        OUTPUTS:
        - y_hat : numpy array of size NxC, where C is the number of classes
        '''
        ################# YOUR CODE HERE ####################
        self.layer[0].a = x@self.layer[0].W.T+self.layer[0].b
        self.layer[0].z = MLP.sigmoid(self.layer[0].a)
        for k in range (1, len(self.layer)) :
            self.layer[k].a = (self.layer[k-1].z@self.layer[k].W.T)+self.layer[k].b
            self.layer[k].z = MLP.sigmoid(self.layer[k].a)
        y_hat = self.layer[len(self.layer)-1].z


        ################ END OF YOUR CODE HERE ##############

        return y_hat

  
    
    def loss(y_hat, y) :
        '''
        Compute the loss between y_hat and y! they can be 1D or 2D arrays!

        INPUTS:
        - y_hat : numpy array of size NxC, N number of samples, C number of classes. It contains the estimated values of y
        - y : numpy array of size NxC with one 1 in each row, corresponding to the correct class for that sample

        OUTPUTS:
        - L : MSE loss
        '''
        S=0
        for k in range(len(y_hat)):
            if y_hat[k]>=0.5 :
                y_hat[k]=1
            else :
                y_hat[k] = 0
            S=S+(y_hat[k]!=y[k])  
        return (S/(len(y)))


    
    def accuracy(y_hat,y) :
        '''
        Compute the accuracy between y_hat and y

        INPUTS:
        - y_hat : numpy array of size NxC, C number of classes. It contains the estimated values of y
        - y : numpy array of size NxC with correct values of y

        OUTPUTS:
        - acc : the accuracy value between 0 and 1
        '''

        S=0
        for k in range(len(y_hat)):
            if (y_hat[k]>=0.5) :
                y_hat[k]=1
            else :
                y_hat[k] = 0
            S+=(y_hat[k]==y[k])


        return(S/len(y))          

 
    
    def backpropagation(self,x,y,y_hat,learning_rate) :
        '''
        Backpropagate the error from last layer to input layer and then update the parameters

        INPUTS:
        - y_hat : numpy array of size NxC, C number of classes. It contains the estimated values of y
        -y : numpy array of size NxC with correct values of y

        OUTPUTS: (compute the error at the different levels and for each layer)
        - d_a
        - d_z
        - delta_L
        - delta_l
        - d_W
        - d_b
        '''
        # compute gradients
        #last layer
        laym1 = len(self.layer)-1
        self.layer[laym1].d_z = y_hat-y
        delta_L = self.layer[laym1].d_z 
        self.layer[laym1].d_a = np.multiply(self.layer[laym1].d_z, MLP.d_sigmoid(self.layer[laym1].a))

        #other layers
        for k in range (laym1-1,1,-1):
            self.layer[k].d_z = self.layer[k+1].d_a* self.layer[k+1].W
            self.layer[k].d_a = np.multiply(self.layer[k].d_z, MLP.d_sigmoid(self.layer[k].a))
            self.layer[k].d_b = self.layer[k].d_a
            self.layer[k].d_W = self.layer[k].d_a @ self.layer[k-1].z.T

        #first layer

        self.layer[0].d_z =  self.layer[1].d_a@self.layer[1].W
        self.layer[0].d_a = np.multiply(self.layer[0].d_z, MLP.d_sigmoid(self.layer[0].a))
        self.layer[0].d_b = self.layer[0].d_a
        self.layer[0].d_W = self.layer[0].d_a.T@x

    # apply gradients
        # just one for loop passing through all layers is sufficient
        # apply the gradients only to self.layer[i].b and self.layer[i].W


        for k in range (len(self.layer)) :
            self.layer[k].W -= learning_rate*self.layer[k].d_W
            self.layer[k].b -= learning_rate*self.layer[k].d_b

        
    def training(self,x,y,learning_rate,num_epochs,verbose=False, print_every_k=1) :
        '''
        Training your network
        
        INPUTS:
        - x : numpy array of size NxD, D number of features of your input
        - y : numpy array of size NxC, C number of classes, with correct values of target
        - learning_rate : a numpy scalar containing your learning rate
        - num_epochs : a numpy scalar representing the number of epochs with which train your networks
        - verbose : a boolean False by default, if True print the training loss and training accuracy values
                    if False only store them
        - print_every_k : a numpy scalar equal 1 by default, if verbose is True print the result every print_every_k epochs

        OUTPUTS: /
        '''
        accuracy=[]
        loss=[]

        # iterate for num_epochs number of epochs
        for epoch in range(num_epochs) :
            
            # shuffle your training set
            shuffle=np.random.permutation(range(x.shape[0]))
            x_shuffled=x[shuffle]
            y_shuffled=y[shuffle]
            
            # sample by sample forward and backward through the network using stochastic gradient descent (SGD)
            for sample in range(x.shape[0]) :
                y_hat=self.forward(x_shuffled[sample])
                self.backpropagation(x_shuffled[sample].reshape(1,x.shape[1]),y_shuffled[sample],y_hat,learning_rate)

            # check how is performing the network after each epoch
            # estimate the training labels
            Y_hat=self.forward(x)
            # compute the loss
            loss.append(MLP.loss(Y_hat,y))
            # compute the accuracy
            accuracy.append(MLP.accuracy(Y_hat,y))
            
            # if verbose is True print the results every print_every_k
            if ((verbose == True) and (epoch%print_every_k==0)):
                print('Epoch %d : loss = %.5e, accuracy = %.2f %%' %(epoch,loss[epoch],100*accuracy[epoch]))

        self.losses=loss
        self.accuracies=accuracy
            