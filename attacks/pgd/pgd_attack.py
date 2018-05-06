"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from utils.squeeze import reduce_precision_py
from robustness.feature_squeezing import FeatureSqueezingRC

l2_dist = lambda x1,x2: np.sum((x1-x2)**2, axis=tuple(range(len(x1.shape))[1:]))

# Optimizes for 3 models for now, could be extended into more.
class CombinedLinfPGDAttack:
  def __init__(self, model1, model2, model3, epsilon, k, a, random_start, loss_func, Y=None,
               sq1 = lambda x:x, sq2 = lambda x:x, sq3 = lambda x:x):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point.
       This simulatenously runs the Combined Linf attack for 4 models
       """
    self.model1 = model1
    self.model2 = model2
    self.model3 = model3
    self.epsilon = epsilon
    self.vanilla_model = model1   # Models 1,2 are bit-depth models 
	
    self.k = k
    self.a = a
    self.Y = np.argmax(Y, axis = 1) # Target Labels
    self.rand = random_start

    self.sq1 = sq1
    self.sq2 = sq2
    self.sq3 = sq3
    
    vanilla_y_softmax = tf.nn.softmax(self.vanilla_model.pre_softmax)
    y1_softmax        = tf.nn.softmax(self.model1.pre_softmax)
    y2_softmax        = tf.nn.softmax(self.model2.pre_softmax)
    y3_softmax        = tf.nn.softmax(self.model3.pre_softmax) 
    t1 = y1_softmax - vanilla_y_softmax
    t2 = y2_softmax - vanilla_y_softmax
    t3 = y3_softmax - vanilla_y_softmax
    diff_softmax_1 = tf.nn.relu(tf.reduce_sum(tf.multiply(t1, t1), axis=1) - 0.8)
    diff_softmax_2 = tf.nn.relu(tf.reduce_sum(tf.multiply(t2, t2), axis=1) - 0.8)
    diff_softmax_3 = tf.nn.relu(tf.reduce_sum(tf.multiply(t3, t3), axis=1) - 0.8)
    self.reg_loss = 1000 * (tf.reduce_sum(diff_softmax_1) + tf.reduce_sum(diff_softmax_2) + tf.reduce_sum(diff_softmax_3)) 
    self.loss = model1.xent + model2.xent + model3.xent - self.reg_loss
    # (TODO) Need to add an regularizaton of some sort.
    if loss_func != 'xent':
      print('Unknown loss function. Defaulting to cross-entropy')

    self.grad = (tf.gradients(self.loss, model1.x_input)[0] + tf.gradients(self.loss, model2.x_input)[0] \
                 + tf.gradients(self.loss, model3.x_input)[0])

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
      x = np.copy(x_nat)
    
    r_min = 100000.00
    max_acc = 0
    x_max = x
    sel = 0
    for i in range(self.k):
      # Performing BPDA and EOT here
      x1 = self.sq1(reduce_precision_py(x, 256))
      x2 = self.sq2(reduce_precision_py(x, 256))
      x3 = self.sq3(reduce_precision_py(x, 256))
 
      grad, l, y_cur1, y_cur2, y_cur3, y_van, r_loss = sess.run([self.grad, self.loss, self.model1.y_pred, self.model2.y_pred,
                                                  self.model3.y_pred,self.vanilla_model.y_pred, self.reg_loss], feed_dict={ self.model1.x_input: x1,
                                                  self.model2.x_input: x2, self.model3.x_input: x3, self.vanilla_model.x_input: x1
                                                  , self.model1.y_input: y, self.model2.y_input: y, self.model3.y_input: y,
						  self.vanilla_model.y_input: y })



      sq1_acc = 1 - np.sum(y_cur1 == self.Y)/(float(len(self.Y)))
      sq2_acc = 1 - np.sum(y_cur2 == self.Y)/(float(len(self.Y)))
      sq3_acc = 1 - np.sum(y_cur3 == self.Y)/(float(len(self.Y)))
      van_acc = 1 - np.sum(y_van  == self.Y)/(float(len(self.Y)))
      min_acc = min(sq1_acc, sq2_acc, sq3_acc) 
      print("Itr: ", i, " Loss: ", l, " Reg Loss: ", r_loss)
      print("  Bit Depth: ", sq1_acc, " Median Depth: ", sq2_acc, " Non local means:", sq3_acc, " Vanilla :", van_acc)
      if min_acc > 0.95 and r_loss < r_min:
        r_min = r_loss
	x_max = np.copy(x)
        sel = i 

      x += self.a * np.sign(grad)
      x = np.clip(x, 0, 1)  # ensure valid pixel range
      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
   
    print(" Selected i:", sel, " Reg Loss:", r_min)
    #x_max = np.clip(x_max, x_nat - self.epsilon, x_nat + self.epsilon)
    return x_max


# Optimizes for 3 models for now, could be extended into more.
class CombinedLinfPGDAttackImageNet:
  def __init__(self, model1, model2, model3, epsilon, k, a, random_start, loss_func, Y=None,
               sq1 = lambda x:x, sq2 = lambda x:x, sq3 = lambda x:x):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point.
       This simulatenously runs the Combined Linf attack for 4 models
       """
    self.model1 = model1
    self.model2 = model2
    self.model3 = model3
    self.epsilon = epsilon
    self.vanilla_model = model1   # Models 1,2 are bit-depth models 
	
    self.k = k
    self.a = a
    self.Y = np.argmax(Y, axis = 1) # Target Labels
    self.rand = random_start

    self.sq1 = sq1
    self.sq2 = sq2
    self.sq3 = sq3
    
    vanilla_y_softmax = tf.nn.softmax(self.vanilla_model.pre_softmax)
    y1_softmax        = tf.nn.softmax(self.model1.pre_softmax)
    y2_softmax        = tf.nn.softmax(self.model2.pre_softmax)
    y3_softmax        = tf.nn.softmax(self.model3.pre_softmax) 
    t1 = y1_softmax - vanilla_y_softmax
    t2 = y2_softmax - vanilla_y_softmax
    t3 = y3_softmax - vanilla_y_softmax
    diff_softmax_1 = tf.nn.relu(tf.reduce_sum(tf.abs(t1), axis=1) - 1.2)
    diff_softmax_2 = tf.nn.relu(tf.reduce_sum(tf.abs(t2), axis=1) - 1.2)
    diff_softmax_3 = tf.nn.relu(tf.reduce_sum(tf.abs(t3), axis=1) - 1.2) 
    max_vec = tf.maximum(tf.maximum(diff_softmax_1, diff_softmax_2), diff_softmax_3)
    self.reg_loss = 1000 * tf.reduce_sum(max_vec)
    self.loss = model1.xent + model2.xent + model3.xent - self.reg_loss
    # (TODO) Need to add an regularizaton of some sort.
    if loss_func != 'xent':
      print('Unknown loss function. Defaulting to cross-entropy')

    self.grad = (tf.gradients(self.loss, model1.x_input)[0] + tf.gradients(self.loss, model2.x_input)[0] \
                 + tf.gradients(self.loss, model3.x_input)[0])

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
      x = np.copy(x_nat)
    
    r_min = 100000.00
    max_acc = 0
    x_max = x
    sel = 0
    for i in range(self.k):
      # Performing BPDA and EOT here
      x1 = self.sq1(reduce_precision_py(x, 256))
      x2 = self.sq2(reduce_precision_py(x, 256))
      x3 = self.sq2(reduce_precision_py(x, 256))
 
      grad, l, y_cur1, y_cur2, y_cur3, y_van, r_loss = sess.run([self.grad, self.loss, self.model1.y_pred, self.model2.y_pred,
                                                  self.model3.y_pred,self.vanilla_model.y_pred, self.reg_loss], feed_dict={ self.model1.x_input: x1,
                                                  self.model2.x_input: x2, self.model3.x_input: x3, self.vanilla_model.x_input: x1
                                                  , self.model1.y_input: y, self.model2.y_input: y, self.model3.y_input: y,
						  self.vanilla_model.y_input: y })



      sq1_acc = 1 - np.sum(y_cur1 == self.Y)/(float(len(self.Y)))
      sq2_acc = 1 - np.sum(y_cur2 == self.Y)/(float(len(self.Y)))
      sq3_acc = 1 - np.sum(y_cur3 == self.Y)/(float(len(self.Y)))
      van_acc = 1 - np.sum(y_van  == self.Y)/(float(len(self.Y)))
      min_acc = min(sq1_acc, sq2_acc, sq3_acc) 
      print("Itr: ", i, " Loss: ", l, " Reg Loss: ", r_loss)
      print("  Bit Depth: ", sq1_acc, " Median Depth: ", sq2_acc, " Non local means:", sq3_acc, " Vanilla :", van_acc)
      if min_acc > 0.95 and r_loss < r_min:
        r_min = r_loss
	x_max = np.copy(x)
        sel = i 

      x += self.a * np.sign(grad)
      x = np.clip(x, 0, 1)  # ensure valid pixel range
      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
   
    print(" Selected i:", sel, " Reg Loss:", r_min)
    #x_max = np.clip(x_max, x_nat - self.epsilon, x_nat + self.epsilon)
    return x_max



# Optimizes for 3 models for now, could be extended into more.
class CombinedLinfPGDAttackCIFAR10:
  def __init__(self, model1, model2, model3, epsilon, k, a, random_start, loss_func, Y=None,
               sq1 = lambda x:x, sq2 = lambda x:x, sq3 = lambda x:x):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point.
       This simulatenously runs the Combined Linf attack for 4 models
       """
    self.model1 = model1
    self.model2 = model2
    self.model3 = model3
    self.epsilon = epsilon
    self.vanilla_model = model1   # Models 1,2 are bit-depth models 
	
    self.k = k
    self.a = a
    self.Y = np.argmax(Y, axis = 1) # Target Labels
    self.rand = random_start

    self.sq1 = sq1
    self.sq2 = sq2
    self.sq3 = sq3
    
    vanilla_y_softmax = tf.nn.softmax(self.vanilla_model.pre_softmax)
    y1_softmax        = tf.nn.softmax(self.model1.pre_softmax)
    y2_softmax        = tf.nn.softmax(self.model2.pre_softmax)
    y3_softmax        = tf.nn.softmax(self.model3.pre_softmax) 
    t1 = y1_softmax - vanilla_y_softmax
    t2 = y2_softmax - vanilla_y_softmax
    t3 = y3_softmax - vanilla_y_softmax
    diff_softmax_1 = tf.nn.relu(tf.reduce_sum(tf.multiply(t1, t1), axis=1) - 1.20)
    diff_softmax_2 = tf.nn.relu(tf.reduce_sum(tf.multiply(t2, t2), axis=1) - 1.20)
    diff_softmax_3 = tf.nn.relu(tf.reduce_sum(tf.multiply(t3, t3), axis=1) - 1.20)
    self.reg_loss = 100 * (tf.reduce_sum(diff_softmax_1) + tf.reduce_sum(diff_softmax_2) + tf.reduce_sum(diff_softmax_3)) 
    self.loss = model1.xent + model2.xent + model3.xent - self.reg_loss
    # (TODO) Need to add an regularizaton of some sort.
    if loss_func != 'xent':
      print('Unknown loss function. Defaulting to cross-entropy')

    self.grad = (tf.gradients(self.loss, model1.x_input)[0] + tf.gradients(self.loss, model2.x_input)[0] \
                 + tf.gradients(self.loss, model3.x_input)[0])

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
      x = np.copy(x_nat)
    
    r_min = 100000.00
    max_acc = 0
    x_max = x
    sel = 0
    for i in range(self.k):
      # Performing BPDA and EOT here
      x1 = self.sq1(reduce_precision_py(x, 256))
      x2 = self.sq2(reduce_precision_py(x, 256))
      x3 = self.sq2(reduce_precision_py(x, 256))
 
      grad, l, y_cur1, y_cur2, y_cur3, y_van, r_loss = sess.run([self.grad, self.loss, self.model1.y_pred, self.model2.y_pred,
                                                  self.model3.y_pred,self.vanilla_model.y_pred, self.reg_loss], feed_dict={ self.model1.x_input: x1,
                                                  self.model2.x_input: x2, self.model3.x_input: x3, self.vanilla_model.x_input: x1
                                                  , self.model1.y_input: y, self.model2.y_input: y, self.model3.y_input: y,
						  self.vanilla_model.y_input: y })



      sq1_acc = 1 - np.sum(y_cur1 == self.Y)/(float(len(self.Y)))
      sq2_acc = 1 - np.sum(y_cur2 == self.Y)/(float(len(self.Y)))
      sq3_acc = 1 - np.sum(y_cur3 == self.Y)/(float(len(self.Y)))
      van_acc = 1 - np.sum(y_van  == self.Y)/(float(len(self.Y)))
      min_acc = min(sq1_acc, sq2_acc, sq3_acc) 
      print("Itr: ", i, " Loss: ", l, " Reg Loss: ", r_loss)
      print("  Bit Depth: ", sq1_acc, " Median Depth: ", sq2_acc, " Non local means:", sq3_acc, " Vanilla :", van_acc)
      if min_acc > 0.95 and r_loss < r_min:
        r_min = r_loss
	x_max = np.copy(x)
        sel = i 

      x += self.a * np.sign(grad)
      x = np.clip(x, 0, 1)  # ensure valid pixel range
      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
   
    print(" Selected i:", sel, " Reg Loss:", r_min)
    #x_max = np.clip(x_max, x_nat - self.epsilon, x_nat + self.epsilon)
    return x_max



class LinfPGDAttack:
  def __init__(self, model, epsilon, k, a, random_start, loss_func, squeezer=lambda x:x, Y=None, vanilla_model = None):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.k = k
    self.a = a
    if vanilla_model is None:
      vanilla_model = model
    self.vanilla_model = vanilla_model

    self.Y = np.argmax(Y, axis = 1) # Target Labels
    self.rand = random_start
    self.squeeze = squeezer     # Squeezer for BPDA


    if loss_func == 'xent':
     
      vanilla_y_softmax = tf.nn.softmax(vanilla_model.pre_softmax)
      y_softmax         = tf.nn.softmax(model.pre_softmax)
      t1 = vanilla_y_softmax - y_softmax
      diff_softmax = tf.nn.relu(tf.reduce_sum(tf.multiply(t1, t1), axis=1) - 1.50)
      self.reg_loss = 100 * tf.reduce_sum(diff_softmax)
      
      # Note: The commented parts are notuseful. Keeping them around for later?
      #diff = self.model.x_input - self.model.x_nat
      #self.reg_loss += 10 * tf.reduce_sum(tf.multiply(diff, diff))
      self.loss = tf.minimum(model.xent, vanilla_model.xent) - self.reg_loss

    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax, axis=1)
      self.loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      self.loss = model.xent
    self.grad = tf.gradients(self.loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
      x = np.copy(x_nat)

    max_acc = 0
    x_max = x_nat
    acc = 0.0
    r_min = 100000.00 # A very large value 
    for i in range(self.k):
      x_r = reduce_precision_py(x, 256)
      p_x = self.squeeze(x_r) # First Reduce precision, then squeeze

      grad, l, y_cur, y_cur_vanilla, r_loss = sess.run([self.grad, self.loss, self.model.y_pred, self.vanilla_model.y_pred,
                                                        self.reg_loss],
                                               feed_dict = { self.model.x_input: p_x, self.model.x_input : x_r,
                                                self.model.y_input: y, self.vanilla_model.y_input : y})
      acc          = 1.0 -  (np.sum(y_cur         == self.Y) / float(len(self.Y)))
      acc_vanilla  = 1.0 -  (np.sum(y_cur_vanilla == self.Y) / float(len(self.Y)))
      if min(acc, acc_vanilla) > max_acc + 0.005 :
        max_acc = min(acc, acc_vanilla)
        x_max = np.copy(x)
      if (abs(min(acc, acc_vanilla) - max_acc) <= 0.005) and (r_loss < r_min + 0.005):
	r_min = r_loss 
	x_max = np.copy(x)
 
      x += self.a * np.sign(grad)
      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 1) # ensure valid pixel range
      print("Itr: ", i, " Loss: ", l, " Accuracy: ", acc, " Vanilla Acc:", acc_vanilla, " Reg Loss:", r_loss)

    return x_max


if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model()
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])
  saver = tf.train.Saver()

  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = mnist.test.images[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))
