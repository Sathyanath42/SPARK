# SPARK
OpenCV Study : This repository acts as an archive for all the code i have developed during my internship at the Sparks foundation. 
*****
System Details: OpenCV 4.4 with Anaconda 4.9.2 and Python 3.7
*****
GRIP TASK 1 : Object Detection with MobileNetSSD
* USAGE : python GRIP_task1.py --image images/sofa.png --prototxt bvlc_googlenet.prototxt --model bvlc_googlenet.caffemodel --labels synset_words.txt
* Built with [MobileNetSSD](https://github.com/PINTO0309/MobileNet-SSD-RealSense/tree/master/caffemodel/MobileNetSSD).
* Using the [Word set](https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt) 
* Complete reference [Object Detection with Deep Learning](https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/)
* [Demo](https://youtu.be/nDqsCxinr50) 
*****
Additional grip task 1 : Detection of objects using Keras
* USAGE: python predict.py --image images/dog.jpg --model output/simple_nn.model --labelb output/simple_nn_lb.pickle --width 32 --height 32 --flatten 1
* Complete reference [Keras tutorial](https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/)
*****
GRIP Task 2 : Traffic Sign Detection with Keras and Tensorflow
* USAGE for running model: GRIP_task2_model.py --dataset gtsrb-german-traffic-sign --model output/trafficsignnet.model
* USAGE for prediction: GRIP_task2_pred.py --model output/trafficsignnet.model --images gtsrb-german-traffic-sign/Test --testimage images
* Built with Keras and Tensorflow
* Complete reference [Traffic sign detection](https://www.pyimagesearch.com/2019/11/04/traffic-sign-classification-with-keras-and-deep-learning/)
* [Demo](https://youtu.be/k0MZAUCY4Fw)

