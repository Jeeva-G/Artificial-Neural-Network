import numpy as np

#Activation Function - Sigmoid
def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)

#Derivative of Activation Function - Sigmoid
def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0-x**2


class MLP:
    ''' Multi-layer perceptron class. '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias)
        self.layers.append(np.ones(self.shape[0]+1))
        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))
        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        #Building dummy matrix with ones..
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))
        # dw will hold last change in weights (for momentum)
        self.dw = [0,]*len(self.weights)
        #print self.dw

        # Reset weights
        self.reset()


    def reset(self):
        ''' Reset weights '''

        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            # Look at this for [...] example --> http://python-reference.readthedocs.org/en/latest/docs/brackets/ellipsis.html
            #Doing this computation (2*Z-1)*0.25 just to keep the values between -0.25 to 0.25
            #Replacing the newly created weight matrix with the random values between -0.25 to 0.25
            self.weights[i][...] = (2*Z-1)*0.25

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer
        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.shape)):
            # Propagate activity
            #As per the algorithm, multiplying the layer input with corresponding layer weights and applying sigmoid function on it.
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))

        # Return output
        return self.layers[-1]


    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        # After calculating the error for output layer, we are multiplying the error of a layer with that layer's value
        # We need the actual value we had before applying activation function and so applying derivative of that function
        # Then multiplying that error with actual value to get delta which is the new weight to be added with old weight
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)
        #print deltas


        # Compute error on hidden layers
        # To come backwards -- > range(len(self.shape)-2,0,-1) For example, 
        # print range(len((3,3,3,4,3,5,6))-2,0,-1)  --> [5, 4, 3, 2, 1]
        # we are coming from backwards to calculate the delta value for each hidden layer from last.
        # To calculate the delta for each hidden layer, we need to multiply the output layer delta with the weight of that
        # hidden layer and acual value before applying activation function on that layer.
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
            # We are adding up the delta in the very starting and so using deltas.insert(0,delta). 
            # We are doing this because we are calculating the delta from backwards and so stacking up 
            # to build the delta matrix. 
            deltas.insert(0,delta)
        #print deltas 
        # Update weights for each layer
        for i in range(len(self.weights)):
        	# Getting data for a particular layer to multiply and make sure it is two dimensional array
            layer = np.atleast_2d(self.layers[i])
            # Getting delta for that particular layer
            delta = np.atleast_2d(deltas[i])
            # Adding up the delta weight with the old weight & multiply it by learning rate(to make sure that the 
            # weight changes according to our learning rate) and momentum to minimize the oscillation..
            dw = np.dot(layer.T,delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw

        # Return error
        return (error**2).sum()

if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    def learn(network,samples, epochs=5000, lrate=.1, momentum=0.1):
        # Train 
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward( samples['input'][n] )
            network.propagate_backward( samples['output'][n], lrate, momentum )
        # Test
        for i in range(samples.size):
            o = network.propagate_forward( samples['input'][i] )
            print i, samples['input'][i], '%.2f' % o[0],
            print '(expected %.2f)' % samples['output'][i]
        print

    network = MLP(2,2,1)
    samples = np.zeros(4, dtype=[('input',  float, 2), ('output', float, 1)])
    learn(network,samples)

# Example 1 : OR logical function
# -------------------------------------------------------------------------
print "Learning the OR logical function"
network.reset()
samples[0] = (0,0), 0
samples[1] = (1,0), 1
samples[2] = (0,1), 1
samples[3] = (1,1), 1
learn(network, samples)


# Example 2 : AND logical function
# -------------------------------------------------------------------------
print "Learning the AND logical function"
network.reset()
samples[0] = (0,0), 0
samples[1] = (1,0), 0
samples[2] = (0,1), 0
samples[3] = (1,1), 1
learn(network, samples)

# Example 3 : XOR logical function
# -------------------------------------------------------------------------
print "Learning the XOR logical function"
network.reset()
samples[0] = (0,0), 0
samples[1] = (1,0), 1
samples[2] = (0,1), 1
samples[3] = (1,1), 0
learn(network, samples)

# Example 4 : Learning sin(x)
# -------------------------------------------------------------------------
print "Learning the sin function"
network = MLP(1,10,1)
samples = np.zeros(500, dtype=[('x',  float, 1), ('y', float, 1)])
samples['x'] = np.linspace(0,1,500)
samples['y'] = np.sin(samples['x']*np.pi)

for i in range(10000):
    n = np.random.randint(samples.size)
    network.propagate_forward(samples['x'][n])
    network.propagate_backward(samples['y'][n])

plt.figure(figsize=(10,5))
# Draw real function
x,y = samples['x'],samples['y']
plt.plot(x,y,color='b',lw=1)
# Draw network approximated function
for i in range(samples.shape[0]):
    y[i] = network.propagate_forward(x[i])
plt.plot(x,y,color='r',lw=3)
plt.axis([0,1,0,1])
plt.show()
