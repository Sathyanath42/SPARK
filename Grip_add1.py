import matplotlib #plotting package, AGG used to save plots to disk
matplotlib.use("Agg")

# import the necessary packages
#scklearn helps binarize labels, splits data for training/testing, generating a training report in our terminal

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt 
import numpy as np
import argparse
import random
import pickle
from cv2 import cv2
import os

#parse the input and output paths
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required = True, help ="input the data set (images")
ap.add_argument("--model", required = True, help ="output model path")
ap.add_argument("--labelb", required= True, help= "output label binariser path")
ap.add_argument("--plot", required = True, help = "output accuracy plot path")
args = vars(ap.parse_args())

#initialise lists for the data and the labels which turn into numpy arrays
print("Loading images")
data = []
labels = []

#image path taken and images shuffled
imagepath = sorted(list(paths.list_images(args["dataset"])))
random.seed(42) #same seed shuffles randomly the same way
random.shuffle(imagepath)

#loop over all the input images
for imp in imagepath:
#load image, resize it to 32X32 pixels and flatten image
	image = cv2.imread(imp)
	image = cv2.resize(image, (32,32)).flatten()
	data.append(image) #append all images into data after processing
	label = imp.split(os.path.sep)[-2]
	labels.append(label)

#converting to nparray and scaling pixel intensities
data = np.array(data, dtype = "float") / 255.0
labels = np.array(labels)

(trainx, testx, trainy, testy) = train_test_split(data, labels, test_size = 0.25, random_state=42)

lb = LabelBinarizer() 
#hot-encoding used to binarisation of the data
#find all the unique labels and transforms them into one-hot encoded labels
trainy = lb.fit_transform(trainy) 
testy = lb.transform(testy)

#define the network model with one input layer, 2 hidden layers and one output layer
model = Sequential()
#input layer and first hidden layer, 
#input_shape of 3072 = 32X32X3 , first layer 1024
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
#second layer
model.add(Dense(512, activation="sigmoid"))
#ouput layer , 3 nodes, one for each class label, here 3
model.add(Dense(len(lb.classes_), activation="softmax"))

#####Compiling the model
#initialise our learning rate and total number of epochs to train
INIT_LR = 0.01
EPOCHS = 80

print("training the network---")
#stochastic gradient descent optimiser with categorical crossentropy
opt = SGD(lr=INIT_LR) 
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

######Training the neural network
#batch sizes can be increased based on computing power
H = model.fit(x=trainx, y=trainy, validation_data=(testx, testy), epochs=EPOCHS, batch_size=32)


