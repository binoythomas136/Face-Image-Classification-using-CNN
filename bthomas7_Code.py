#!/usr/bin/env python
# coding: utf-8

# In[115]:


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import cv2
import numpy as np
from scipy.stats import multivariate_normal
import gc
from decimal import Decimal
import matplotlib.pyplot as plt
tf.logging.set_verbosity(tf.logging.INFO)

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
      logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



I=7500
imageVector = np.empty((15000,28,28))
for i in range(I):
	img = cv2.imread("/Users/binoythomas/Desktop/face_images1/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
  
	image = cv2.cvtColor(im_resized, cv2.COLOR_BGR2GRAY)
	imageVector[i][:][:] = image
	#plt.imshow(image,cmap='gray')   
	#plt.show()



for i in range(I):
	img = cv2.imread("/Users/binoythomas/Desktop/nonfaces/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	imageVector[7500+i][:][:] = image
	#plt.imshow(image)   
	#plt.show()

print(np.shape(imageVector))

imageVector = imageVector - np.mean(imageVector, axis=0)

imageVector = imageVector / np.std(imageVector, axis=0)

imageLabels = np.empty((15000))
for i in range(I):
    imageLabels[i]=1


# In[124]:


for i in range(I):
    imageLabels[7500+i]=0


# In[126]:




# In[127]:


a = imageVector
b = imageLabels
ab=shuffle_in_unison(a, b)


# In[128]:


train_data=ab[0]
train_labels=ab[1]


# In[130]:


train_data = train_data/np.float32(255)
train_labels = train_labels.astype(np.int32) 


# In[131]:


I=250
testVector = np.empty((500,28,28))
for i in range(I):
	img = cv2.imread("/Users/binoythomas/Desktop/face_images1/"+str(i+7500)+str(1)+".jpg")
	im_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	testVector[i][:][:] = image
    


for i in range(I):
	img = cv2.imread("/Users/binoythomas/Desktop/non face images/"+str(i+7500)+str(1)+".jpg")
	im_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	testVector[250+i][:][:] = image

testLabels = np.empty((500))
for i in range(250):
    testLabels[i]=1
for i in range(250):
    testLabels[250+i]=0    

eval_data = testVector/np.float32(255)
eval_labels = testLabels.astype(np.int32)


# In[132]:


mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model3")


# In[133]:


# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)


# In[134]:


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=10,
    num_epochs=None,
    shuffle=True)


# In[135]:


mnist_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])


# In[ ]:


eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=10,
    shuffle=False)

eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
