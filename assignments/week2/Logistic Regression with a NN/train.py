import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Load dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes =load_dataset()
print("dataset loaded successfully !")

#Let's visualize some images from the dataset
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

# train_set_x_orig.shape = train_set_y =(209,64,64,3)
#209 is the number of trainning example
# test_set_x_orig.shape = test_set_y.shape = (50,64,64,3)

#Let's reshape and standardize data such that each example is now a vector of size (64*64*3,1)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

#Let's code the sigmoid function (Our activation function here)
def sigmoid(z):
	return 1/(1+np.exp(-z))
def log(a):
	return np.log(a)
#Initializing parameters
def initialize_with_zeros(dim):
	b = 0
	w = np.zeros((dim,1))
	assert(w.shape == (dim,1))
	assert(isinstance(b,float) or isinstance(b,int))
	return w,b

## Forward and Backward propagation
def propagate(w,b,X,Y):
	#We compute the gradient and the cost here
	m = X.shape[1] 
	z = np.dot(w.T,X)+b
	A = sigmoid(z)
	#Derivatives
	dz = A - Y
	dw = np.dot(X,dz.T)/m
	db = np.sum(dz)/m
	gradient = {
		"dw" : dw,
		"db" : db
	}
	cost = -np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))/m
	print("dw.shape:",dw.shape)
	print("w.shape:",w.shape)
	# assert(dw.shape == w.shape)
	# assert(db.dtype == float)
	cost = np.squeeze(cost)
	assert(cost.shape == ())

	return gradient,cost
## Optimization
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
	
	costs = []
	for i in range(num_iterations):
		gradient,cost = propagate(w,b,X,Y)
		dw = gradient["dw"]
		db = gradient["db"]

		w = w - learning_rate * dw
		b = w - learning_rate * db
		if i%100 == 0 : 
			costs.append(cost)
		if print_cost and i%100==0:
			print ("Cost after iteration %i: %f" %(i, cost))
	params = {
		"w" : w,
		"b" : b
	}

	gradient ={
		"dw" : dw,
		"db" : db
	}
	return params,gradient,costs
#Prediction function
def predict(w,b,X):
	m = X.shape[1]
	Y_prediction = np.zeros((1,m))
	w = w.reshape(X.shape[0],1)
	A = sigmoid(np.dot(w.T,X)+b)
	Y_prediction = np.where(A<0.5,0.0,1.0)
	assert(Y_prediction.shape==(1,m))
	return Y_prediction
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
	w,b = initialize_with_zeros(X_train.shape[0])
	params,gradient,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost=True)
	w = params["w"]
	b = params["b"]
	Y_prediction_test = predict(w,b,X_test)
	Y_prediction_train = predict(w,b,X_train)
	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
	d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
	return d