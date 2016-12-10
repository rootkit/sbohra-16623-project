// Copyright 2016 Sudev Bohra All rights reserved.
//
// Created by Sudev Bohra on 12/10/16.
//
// Not for commercial use.

# This file is used to convert data.npy you get from Caffe-To-Tensorflow repo's 
# code. This also freezes the constants so that the weights are baked into
# the graph protobuf. Ready for transport to any platform
 

import numpy as np
from model import DeepFace_set003_net
import tensorflow as tf


#np.set_printoptions(threshold=np.nan)

def load_image(image_path, is_jpeg):
        # Read the file
        file_data = tf.read_file(image_path)
        # Decode the image data
        img = tf.image.decode_jpeg(file_data, channels=3)
        print img.get_shape()
	img = tf.image.rgb_to_grayscale(img, name=None)
	return img

#img = load_image("./face-aligned-ex.jpg", True)
#img = tf.cast(img, tf.float32)
#img = tf.div(img,255)
#img = tf.reshape(img, (1, 128, 128, 1))
#print img.get_shape()
input_node = tf.placeholder(tf.float32,shape=(1, 128, 128, 1))
from tensorflow.python.framework import graph_util
net = DeepFace_set003_net({'data': input_node})
with tf.Session() as sess:
    # Load the data
    sess.run(tf.global_variables_initializer())
    net.load('data.npy', sess)
    graph_def = sess.graph.as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ["mfm_fc1"])
    with tf.gfile.GFile("./frozen_face.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())
    #saver = tf.train.Saver(sharded = False)
    #saver.save(sess, 'face-chkpt', latest_filename='chkpt_state')
    #tf.train.write_graph(sess.graph.as_graph_def(), './', 'face.pb.txt')
    
    # Forward pass
    #print [n.name for n in tf.get_default_graph().as_graph_def().node]
    #print "OUTPUT IS",net.get_output()
    #embeddings = tf.get_default_graph().get_tensor_by_name("mfm_fc1:0")

    #output = sesh.run(embeddings)
    #print output
