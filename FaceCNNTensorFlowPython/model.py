// Copyright 2016 Sudev Bohra All rights reserved.
//
// Created by Sudev Bohra on 12/10/16.
//
// Not for commercial use.

from network import Network

class DeepFace_set003_net(Network):
    def setup(self):
        (self.feed('data')
             .conv(5, 5, 96, 1, 1, relu=False, name='conv1')
             .mfm(name='mfm1')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(1, 1, 96, 1, 1, relu=False, name='conv2a')
             .mfm(name='mfm2a')
             .conv(3, 3, 192, 1, 1, relu=False, name='conv2')
             .mfm(name='mfm2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(1, 1, 192, 1, 1, relu=False, name='conv3a')
             .mfm(name='mfm3a')
             .conv(3, 3, 384, 1, 1, relu=False, name='conv3')
             .mfm(name='mfm3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(1, 1, 384, 1, 1, relu=False, name='conv4a')
             .mfm(name='mfm4a')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv4')
             .mfm(name='mfm4')
             .conv(1, 1, 256, 1, 1, relu=False, name='conv5a')
             .mfm(name='mfm5a')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv5')
             .mfm(name='mfm5')
             .max_pool(2, 2, 2, 2, name='pool4')
             .fc(512, relu=False, name='fc1')
             .mfm(name='mfm_fc1')
             .fc(99891, relu=False, name='fc2')
             .softmax(name='softmax'))

