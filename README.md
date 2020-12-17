# SPARK
OpenCV Study : This repository acts as an archive for all the code i have developed during my internship at the Sparks foundation. 
*****
System Details: OpenCV 4.4 with Anaconda 4.9.2 and Python 3.7
*****
GRIP TASK 1
* USAGE : python GRIP_task1.py --image images/sofa.png --prototxt bvlc_googlenet.prototxt --model bvlc_googlenet.caffemodel --labels synset_words.txt
* Built with [MobileNetSSD](https://github.com/PINTO0309/MobileNet-SSD-RealSense/tree/master/caffemodel/MobileNetSSD).
* Using the [Word set](https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt) 
* Complete reference [Object Detection with Deep Learning](https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/)
*****
Additional grip task 1
* USAGE: python predict.py --image images/dog.jpg --model output/simple_nn.model --labelb output/simple_nn_lb.pickle --width 32 --height 32 --flatten 1
