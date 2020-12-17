
from tensorflow.keras.models import load_model
import argparse
import pickle
from cv2 import cv2

#parse the input and output paths
ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="path for input image")
ap.add_argument("--model", required=True, help="path to the model")
ap.add_argument("--labelb", required=True, help="path to label binariser")
ap.add_argument("--width", type=int, default=28, help="target spatial dimension width")
ap.add_argument("--height", type=int, default=28, help="target spatial dimension height")
ap.add_argument("--flatten", type=int, default=-1, help="flatten image or not")
args = vars(ap.parse_args())

#load the input image and resize it the required dim
image = cv2.imread(args["image"])
output = image.copy()
image = cv2.resize(image, (args["width"],args["height"]))

#scale the pixel values to [0,1]
image = image.astype("float")/255.0

#check to flatten or not
if args["flatten"]>0:
	image=image.flatten()
	image=image.reshape((1,image.shape[0]))

else:
	image=image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

#load model and binarizer and make prediction
print("loading model and label binarizer")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelb"],"rb").read())

#make prediction on image
preds = model.predict(image)

#class label index with largest probabiliy
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(255, 0, 0), 2)
# show the output image
cv2.imshow("Image", output)
cv2.waitKey(0)
