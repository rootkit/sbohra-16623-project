# sbohra-16623-project
Deep Face Verification on iOS - Sudev Bohra

### Commercial use not allowed. ###

## Introduction:
For my project I use a deep network to run facial verification/recognition on live video on an iOS device.

IIn the last few years, Convolution Neural Network’s (CNN) have proven to be great at various computer vision tasks like object recognition, pose estimation and face recognition. In this paper, I use a light CNN to implement face recognition on an iOS device. 

Systems like DeepFace [2] and FaceNet [3] have beat human-level performance accuracies of over 98-99% on the LFW face recognition. However, these networks are very deep and contain many parameters. DeepFace contains 120 million parmeters and 9 layers. FaceNet contains between 8-20 million parameters and between 9 and 22 layers.

Wu et al.[1] propose a light CNN framework for deep face representation from noisy labels. They define a Max-Feature-Map activation function to replace the, usual ReLU to suppress low-activations. The Max-Feature-Map helps to separate out noisy labels, allowing you to use large automatically collected datasets for training. The CNN This light CNN outputs a 256D CNN based feature vector. Cosine-similarity between these 256D vectors can be used to detect similarity between faces. This representation is a lot more compact compared to systems like DeepFace that extracted 4096D feature vectors. Wu et al.’s CNN achieves, state of the art results of the LFW benchmark of a 98.80% accuracy.

Due to Wu et al. CNN’s light nature and compact face representation, this paper is a great candidate for mobile implementation. I want to see how fast you can run this facial recognition neural network on a mobile device.


## Background:
I use Wu et al.’s CNN framework and the pre-trained model from his paper for this iOS implementation since I wanted to focus on the mobile implementation and not on training the model (which is usually done on a GPU server).

I use Google TensorFlow (TF) as my neural network framework of choice. Since TF is available on all platforms, the models are portable across desktop and mobile. TF represents neural networks as a graph of computations. The model can be written in Python and then the graphs can be run on either C++ or Python. The iOS implementation of TF uses Eigen under the hood, which supports explicit vectorization for ARM NEON. 

I also explore using Apple’s Metal framework for GPU neural network computations. Although I ran into some issues due to floating point conversions that didn’t allow me to demo the model. I have included the speed of the Metal implementation in this paper as anecdotal evidence.

Initially, I used OpenCV and dlib for facial landmark alignment, which is a necessary step to crop and align faces before passing it to the CNN. Dlib face alignment is based on an ensemble of regression trees and is decently fast. In the end, I replaced this with iOS CoreImage since it runs face detection and 3 point landmark detection very quickly.


I was able to run the network at 5-6 FPS on iOS TensorFlow framework (CPU) and was able to run the network at 10 FPS on Apple Metal framework (GPU).

## Schedule:

✔ Get a Neural Network framework running on iOS. Find dataset and sanitize it.

✔ Build the neural network necessary to extract deep face features based on research paper and train it either on a GPU server

✔ Run the neural network on the iOS device and begin running it on camera. Wire up camera to neural network, detecting face, cropping face and passing to the network and converting response to visual overlay on camera view.

✔ Tweak and build a demo project that uses the deep face features on mobile for face verification/recognition.

Recently, there has been a lot of breakthrough in the use of deep networks in face recognition and face verification tasks. For my project I use a deep network to run facial recognition on live video. I use techniques from the lightened CNN paper and tensorflow framework, along with CoreImage Face detection to run face detection live.


I built a lightened CNN with Max-Feature-Map activations for the face detection in tensorflow.


Other github projects referenced:

VGGNet on Metal iOS:
https://github.com/hollance/VGGNet-Metal/tree/master/VGGNet-iOS/VGGNet

Caffe-to-Tensorflow model conversion:
https://github.com/ethereon/caffe-tensorflow

Tensorflow iOS examples:
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/ios_examples

These are research projects that I heavily referenced:
Research Papers:
[1403.2802] Learning Deep Face Representation - arXiv.org
https://arxiv.org › cs
by H Fan - ‎2014 - ‎Cited by 47 - ‎Related articles


A Lightened CNN for Deep Face Representation
https://arxiv.org › cs
by X Wu - ‎2015 - ‎Cited by 12 - ‎Related articles


