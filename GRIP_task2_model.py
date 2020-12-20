import matplotlib
matplotlib.use("Agg")

from trafficmodel.trafficsignnet import trafficsignnet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io
import matplotlib.pyplot as plt 
import numpy as np
import argparse
import random
import os

def load_split(basePath, csvPath):
    data=[]
    labels=[]
#load the csv file, remove the first line(header), shuffle the rows, remove spaces
    rows=open(csvPath).read().strip().split("\n")[1:]
    random.shuffle(rows)

    for (i,row) in enumerate(rows):
        if i>0 and i%1000 == 0:
            print("[INFO] processed {} total images".format(i))
        #split the row into components classID and path
        (label, imagePath) = row.strip().split(",")[-2:]

        #derive the full path to the imagefile and load it
        imagePath = os.path.sep.join([basePath, imagePath])
        image = io.imread(imagePath)
        image = transform.resize(image,(32,32))
        image = exposure.equalize_adapthist(image, clip_limit=0.1) #CLAHE image correction

        #update the data and labels
        data.append(image)
        labels.append(label)

    #convert the dat aand labels into arrays
    data=np.array(data)
    labels=np.array(labels)

    return(data,labels)

#parsing the arguments
ap =argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, help="path to the traffic signal dataset")
ap.add_argument("--model", required=True, help="path to the output model")
ap.add_argument("--plot", required=True, help="path to the training history plot")
args = vars(ap.parse_args())

#initialise the number of epohcs to train for and the base learning rate and batch size
NUM_EPOCHS = 30
INIT_LR = 1e-3
BS = 64

#load the label names
labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

#derive the paths to the training and testing files
trainPath = os.path.sep.join([args["dataset"], "Train.csv"])
testPath = os.path.sep.join([args["dataset"], "Test.csv"])

#load the training and testing data
print("[INFO] Loading the training and testing data..")
(trainX, trainY) = load_split(args["dataset"], trainPath)
(testX, testY) = load_split(args["dataset"], testPath)

#scaling the data to the range of [0,1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# one-hot encode the training and testing labels
numLabels = len(np.unique(trainY))
trainY = to_categorical(trainY, numLabels)
testY = to_categorical(testY, numLabels)

# account for skew in the labeled data
classTotals = trainY.sum(axis=0)
classWeight = classTotals.max() / classTotals

# construct the image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	vertical_flip=False,
	fill_mode="nearest")

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model = trafficsignnet.build(width=32, height=32, depth=3,
	classes=numLabels)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=trainX.shape[0] // BS,
	epochs=NUM_EPOCHS,
	class_weight=classWeight,
	verbose=1)
