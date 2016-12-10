# sbohra-16623-project
Deep Face Verification on iOS - Sudev Bohra

Commercial use not allowed.

## Background:
For my project I would like to use a deep network to run facial verification/recognition on live video on an iOS device.

## The Challenge:
This project is challenging since running deep networks for a real-world task involves building a neural network that is not too large,
especially if you are trying to run the network in real-time front-facing camera on a mobile phone. By doing this project I hope
to learn how to efficiently use computer vision and deep learning on a mobile device to create an experience.

## Goals & Deliverables
Plan to achieve: Implement a deep face feature extractor that can be used to reasonably verify if two faces have the same identity. I want
to run this model on an iOS device front camera.

Success would be a mobile app that works in recognizing real people using live camera. I will evaluate how well the detector does by running both 
examples from datasets and estimating what the precision and recall of the system is in real life photo-to-person matching.

## Schedule:
Week 1:
Get a Neural Network framework running on iOS. Find dataset and sanitize it.
Week 2:
Build the neural network necessary to extract deep face features based on research paper and train it either on iOS or on a GPU
server
Week 3:
Run the neural network on the iOS device and begin running it on camera. Wire up camera to neural network, detecting face, cropping face 
and passing to the network and converting response to visual overlay on camera view.
Week 4:
Tweak and build a demo project that uses the deep face features on mobile for face verification/recognition.

Recently, there has been a lot of breakthrough in the use of deep networks in face recognition and face verification tasks. For my project I would like to use a deep network to run facial recognition on live video. I will be using techniques from the lightened CNN paper and tensor flow, along with CoreImage Face detection to run face detection live.


I will be building a Pyramid CNN for the face detection. The data set I will be using will be Faces In The Wild dataset which is also used in the paper. I will be trying to make the performance of this network as fast as possible and try to reduce the size of the model without hurting accuracy of the algorithm.


I will be using the following two research papers for my project:

Research Papers:
[1403.2802] Learning Deep Face Representation - arXiv.org
https://arxiv.org › cs
by H Fan - ‎2014 - ‎Cited by 47 - ‎Related articles


A Lightened CNN for Deep Face Representation
https://arxiv.org › cs
by X Wu - ‎2015 - ‎Cited by 12 - ‎Related articles


