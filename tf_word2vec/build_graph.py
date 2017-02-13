# [url](https://www.tensorflow.org/versions/master/tutorials/word2vec/)
#
# Author: Yi Zhang <beingzy@gmail.com> 
# Date: 2017/02/12
import os
import math 
import numpy as np 
import pandas as pd 
import tensorflow as tf 


# =============
# parameter configuration
# =============
batch_size = 32 # normally between 16 and 513
vocabulary_size = 10000
embedding_size = 100 
num_sampled = 1 #

embeddings = tf.Variable(
    tf.random_uniform(vocabulary_size, embedding_size), -1.0, 1.0))

nce_weights = tf.Variable(
	tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))

nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# skip-gram model 
# placeholders for inputs 
train_inputs = tf.placeholders(tf.int32, shape=[batch_size])
train_labels = tf.placeholders(tf.int32, shape=[batch_size, 1])

# retrieve embeddings of the source words
embed = tf.nn.embedding_lookup(embeddings, train_inputs)
# compute the NCE loss, using a sample of the negative labels each time 
loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, num_sampled, vocabulary_size))
# use the SGD optimizer 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
