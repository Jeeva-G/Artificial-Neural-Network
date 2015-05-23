#imports
import numpy as np 

#BackPropogation Class
class BackPropogationNetwork:
	"""A Back - Propogation Network"""


	#Class members
	LayerCount = 0  #It will be a tuple with neuron count in each layer. eg:- (2,2,1) -- 2 nodes in input, 2 in hidden and 1 in output
	Shape = None
	Weights = []

	#Class methods
	def __init__(self,LayerSize):

		#Layer Info
		self.LayerCount = len(LayerSize) - 1  #Number of layers minus 1 as the input layer is not counted.
		self.Shape = LayerSize

		#Input/Output data from last run
		self._LayerInput = []
		self._LayerOutput = []

		#Create Weight Arrays
		#Zip combines the i-th element in both sequence together to make a tuple
		for (l1,l2) in zip(LayerSize[:-1],LayerSize[1:]):
			self.Weights.append(np.random.normal(scale = 0.01,size = (l2,l1+1)))
		#The above calculation generates weights used for multiplication. For (2,2,1), it will be considered as (3,2),(3,1)
		#because one will be added for bias. (3,2) will be considered here as (2,3) because of backpropogation. (2,3) generates
		#two dimensional array, which is two rows and three columns. Now this means, three input(one row) will be multiplied by
		#one hidden node and the other row will be multiplied by another hidden node. Three columns because three input node in the layer.


	#Run Method
	def Run(self, input):
		"""Run the Network based on the input data"""
		lnCases = input.shape[0]

		#Clear out the previous intermediate value lists
		self._LayerInput = []
		self._LayerOutput = []

		#Run it !!!! 
		#If it is for input layer, get the input from "input", it is for hidden layer, get the input from previous layer
		for index in range(self.LayerCount):
			#Determine the layer input
			if index == 0:  #If it is the first or input layer..
				#Multiply the weights with input(input data stacked with bias( which is np.ones([1,lnCases]) -- random ones of same size as input))
				#Stacking ones for bias node - Appending ones with input for bias node
				# In the input ( list or matrix ), each row is considered as an input and it is transposed for matrix multiplication with weights
				#Because matrix multiplication is a product of row vector and column vector
				LayerInput = self.Weights[0].dot(np.vstack([input.T,np.ones([1,lnCases])])) 
			else:
				# self._LayerOutput[-1] takes very last element in the list layer output, in here it takes the output of the layer before this..
				# Because we keep appending the output of a certain layer to Layeroutput, so the last appended element can be taken by using [-1]
				LayerInput = self.Weights[index].dot(np.vstack([self._LayerOutput[-1],np.ones([1,lnCases])]))

			self._LayerInput.append(LayerInput)
			self._LayerOutput.append(self.sgm(LayerInput))

		#Last layer output can be extracted by using [-1], last appended element. And we are transposing it back to the normal structure as input
		return self._LayerOutput[-1].T


	#TrainEpoch Method
	def TrainEpoch(self,input,target,trainingRate = 0.2):
		"""This method trains the network for one Epoch"""
		delta = []
		lnCases = input.Shape[0]

		#First run the network
		self.Run(input)

		#Calculate our deltas
		for index in reversed(range(self.LayerCount)):
			if index == self.LayerCount - 1 :
				#Compare to the target values
				output_delta = self._LayerOutput[index] - target.T
				error = np.sum(output_delta**2)
				delta.append(output_delta * self.sgm(self._LayerInput[index],True))
			else:
				#Compare to the following layer's delta
				delta_pullback = self.weights(index+1).T.dot(delta[-1])
				delta.append(delta_pullback[:-1, :]) * self.sgm(self._LayerInput[index],True))


	#Transfer Functions
	def sgm(self,x,Derivative = False):
		if not Derivative:
			return 1/1-(np.exp(-x))
		else:
			out = self.sgm(x)
			return out*(1-out)


#If run as a script, create a test object

if __name__ == '__main__':
	bpn = BackPropogationNetwork((2,2,1))
	print bpn.Shape
	print bpn.Weights

	lvInput = np.array([[0,0],[1,1],[-1,0.5]])
	lvOutput = bpn.Run(lvInput)

	print "Input: {0}\n Output: {1}".format(lvInput,lvOutput)








