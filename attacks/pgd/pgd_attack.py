"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

import tensorflow as tf
import numpy as np

from utils.squeeze import reduce_precision_py, get_squeezer_by_name 
from robustness.feature_squeezing import FeatureSqueezingRC

l2_dist = lambda x1,x2: np.sum((x1-x2)**2, axis=tuple(range(len(x1.shape))[1:]))

# Debugged Implementations (hopefully)
class CombinedLinfPGDAttack:
  def __init__(self, model_vanilla, model_bit, model_median, model_local, epsilon, k, a, random_start, loss_func, Y,
                sq_bit, sq_median, sq_local):

    self.model_vanilla = model_vanilla
    self.model_bit = model_bit
    self.model_median = model_median
    self.model_local = model_local

    # (TODO) : Substitute these with Flags
    # (TODO) : Add a regularizer
    THRESHOLD = FLAGS.threshold
    REG_LAMBDA = FLAGS.reg_lambda

    print("Threshold: %.3f  Regularization: %.3f" % (THRESHOLD, REG_LAMBDA))

    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.Y = np.argmax(Y, axis = 1) # Target Labels
    self.rand = random_start

    self.sq_bit = sq_bit
    self.sq_median = sq_median
    self.sq_local = sq_local
 
    vanilla_softmax = tf.nn.softmax(self.model_vanilla.pre_softmax)
    bit_softmax     = tf.nn.softmax(self.model_bit.pre_softmax)
    median_softmax  = tf.nn.softmax(self.model_median.pre_softmax)
    local_softmax   = tf.nn.softmax(self.model_local.pre_softmax)

    bit_t    = bit_softmax - vanilla_softmax
    median_t = median_softmax - vanilla_softmax
    local_t  = local_softmax - vanilla_softmax

    diff_bit = tf.nn.relu(tf.reduce_sum(tf.multiply(bit_t, bit_t), axis=1) - THRESHOLD)
    diff_median = tf.nn.relu(tf.reduce_sum(tf.multiply(median_t, median_t), axis=1) -  THRESHOLD)
    diff_local = tf.nn.relu(tf.reduce_sum(tf.multiply(local_t, local_t), axis=1) - THRESHOLD)

    self.reg_loss = REG_LAMBDA * (tf.reduce_sum(diff_bit) + tf.reduce_sum(diff_median) + tf.reduce_sum(diff_local))
    self.loss = self.model_bit.xent + self.model_median.xent + self.model_local.xent - self.reg_loss

    if loss_func != 'xent':
      print('Unknown loss function. Defaulting to cross-entropy')

    self.grad = (tf.gradients(self.loss, self.model_bit.x_input)[0]  \
                 + tf.gradients(self.loss, self.model_median.x_input)[0] \
                 + tf.gradients(self.loss, self.model_local.x_input)[0])

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
      x_van     = reduce_precision_py(x, 256)
      x_bit     = self.sq_bit(x_van)
      x_median  = self.sq_median(x_van)
      x_local   = self.sq_local(x_van)


      grad, loss, y_vanilla, y_bit, y_median, y_local, reg_loss = sess.run([ self.grad, self.loss,
                  self.model_vanilla.y_pred, self.model_bit.y_pred, self.model_median.y_pred, self.model_local.y_pred,
                  self.reg_loss], feed_dict = { self.model_vanilla.x_input : x_van, self.model_vanilla.y_input : y,
                                                self.model_bit.x_input : x_bit, self.model_bit.y_input : y,
                                                self.model_median.x_input: x_median, self.model_median.y_input:y,
                                                self.model_local.x_input: x_local, self.model_local.y_input:y })


      vanilla_accuracy = 1 - np.sum(y_vanilla == self.Y)/(float(len(self.Y)))
      bit_accuracy     = 1 - np.sum(y_bit == self.Y)/(float(len(self.Y)))
      median_accuracy  = 1 - np.sum(y_median == self.Y)/(float(len(self.Y)))
      local_accuracy   = 1 - np.sum(y_local  == self.Y)/(float(len(self.Y)))
      min_accuracy = min(bit_accuracy, median_accuracy, local_accuracy)

      print("======  Itr: ", i, " Loss: ", loss, " Reg Loss: ", reg_loss)
      output_template = ('Vanilla : ({:.3f}%)   Median: ({:.3f}%)  Bit:({:.3f}%)   Non Local:({:.3f}%)')
      print(output_template.format(vanilla_accuracy, median_accuracy, bit_accuracy, local_accuracy))

      if min_accuracy > max_acc or (min_accuracy >= max_acc and reg_loss < r_min):
        max_acc = min_accuracy
        x_max = np.copy(x)
        sel = i
        r_min = reg_loss


      x += self.a * np.sign(grad)
      x = np.clip(x, 0, 1)  # ensure valid pixel range
      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)

    x_max = np.clip(x_max, x_nat - self.epsilon, x_nat + self.epsilon)

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
    self.sq3 = get_squeezer_by_name('non_local_means_color_13_3_2', 'python') 
    
    vanilla_y_softmax = tf.nn.softmax(self.vanilla_model.pre_softmax)
    y1_softmax        = tf.nn.softmax(self.model1.pre_softmax)
    y2_softmax        = tf.nn.softmax(self.model2.pre_softmax)
    y3_softmax        = tf.nn.softmax(self.model3.pre_softmax) 
    t1 = y1_softmax - vanilla_y_softmax
    t2 = y2_softmax - vanilla_y_softmax
    t3 = y3_softmax - vanilla_y_softmax
    diff_softmax_1 = tf.nn.relu(tf.reduce_sum(tf.abs(t1), axis=1) - 1.4)
    diff_softmax_2 = tf.nn.relu(tf.reduce_sum(tf.abs(t2), axis=1) - 1.4)
    diff_softmax_3 = tf.nn.relu(tf.reduce_sum(tf.abs(t3), axis=1) - 1.4) 
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
      x3 = self.sq3(reduce_precision_py(x, 256))
      x_van = reduce_precision_py(x, 256)
      grad, l, y_cur1, y_cur2, y_cur3, y_van, r_loss = sess.run([self.grad, self.loss, self.model1.y_pred,
                                                                 self.model2.y_pred, self.model3.y_pred,
                                                                 self.vanilla_model.y_pred, self.reg_loss],
                                                                feed_dict={self.model1.x_input: x1,
                                                                           self.model2.x_input: x2,
                                                                           self.model3.x_input: x3,
                                                                           self.vanilla_model.x_input: x_van,
                                                                           self.model1.y_input: y, self.model2.y_input:
                                                                             y, self.model3.y_input: y,
                                                                           self.vanilla_model.y_input: y})

      sq1_acc = 1 - np.sum(y_cur1 == self.Y) / (float(len(self.Y)))
      sq2_acc = 1 - np.sum(y_cur2 == self.Y) / (float(len(self.Y)))
      sq3_acc = 1 - np.sum(y_cur3 == self.Y) / (float(len(self.Y)))
      van_acc = 1 - np.sum(y_van == self.Y) / (float(len(self.Y)))
      min_acc = min(sq1_acc, sq2_acc, sq3_acc)

      rc = FeatureSqueezingRC(self.vanilla_model.keras_model, "FeatureSqueezing?squeezer=non_local_means_color_13_3_2")
      y_rc = np.argmax(rc.predict(x_van), axis=1)
      rc_acc = 1 - np.sum(y_rc == self.Y)/ (float(len(self.Y)))
      print("Itr: ", i, " Loss: ", l, " Reg Loss: ", r_loss, "----")
      print("  Bit Depth: ", sq1_acc, " Median Depth: ", sq2_acc, " Non local means:", sq3_acc, " Vanilla :", van_acc,
              "RC acc: ", rc_acc)
      if min_acc > 0.85 and r_loss < r_min:
        r_min = r_loss
        x_max = np.copy(x)
        sel = i

      x += self.a * np.sign(grad)
      x = np.clip(x, 0, 1)  # ensure valid pixel range
      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)

    print(" Selected i:", sel, " Reg Loss:", r_min)
    # x_max = np.clip(x_max, x_nat - self.epsilon, x_nat + self.epsilon)
    return x_max

# Optimizes for 3 models for now, could be extended into more.
class CombinedLinfPGDAttackCIFAR10OLD:
  def __init__(self, model_vanilla, model1, model2, model3, epsilon, k, a, random_start, loss_func, Y,
               sq1, sq2 , sq3):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point.
       This simulatenously runs the Combined Linf attack for 4 models
       """
    self.model1 = model1
    self.model2 = model2
    self.model3 = model3
    self.epsilon = epsilon
    self.vanilla_model = model_vanilla

    self.k = k
    self.a = a
    self.Y = np.argmax(Y, axis = 1) # Target Labels
    self.rand = random_start

    self.sq1 = sq1
    self.sq2 = sq2  # This is just identity
    self.sq3 = sq3

    self.rc1 = FeatureSqueezingRC(self.vanilla_model.keras_model, "FeatureSqueezing?squeezer=bit_depth_5")
    self.rc2 = FeatureSqueezingRC(self.vanilla_model.keras_model, "FeatureSqueezing?squeezer=median_filter_2_2")
    self.rc3 = FeatureSqueezingRC(self.vanilla_model.keras_model, "FeatureSqueezing?squeezer=non_local_means_color_13_3_2")

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
      x_van = reduce_precision_py(x, 256)
      x1 = self.sq1(x_van)
      x2 = self.sq2(x_van)
      x3 = self.sq3(x_van)

      if np.array_equal(x1, x3) == True or np.array_equal(x1, x_van)== True:
        print(" Problem with the squeezer")

      grad, l, y_cur1, y_cur2, y_cur3, y_van, r_loss = sess.run([self.grad, self.loss, self.model1.y_pred,
                                                                 self.model2.y_pred, self.model3.y_pred,
                                                                 self.vanilla_model.y_pred, self.reg_loss],
                                                                feed_dict={self.model1.x_input: x1,
                                                                           self.model2.x_input: x2,
                                                                           self.model3.x_input: x3,
                                                                           self.vanilla_model.x_input: x_van,
                                                                           self.model1.y_input: y, self.model2.y_input:
                                                                             y, self.model3.y_input: y,
                                                                           self.vanilla_model.y_input: y})

      van_bit = sess.run([self.vanilla_model.y_pred], feed_dict={self.vanilla_model.x_input: x1,
                                                                           self.vanilla_model.y_input: y})
      van_local = sess.run([self.vanilla_model.y_pred], feed_dict={self.vanilla_model.x_input: x3,
                                                                 self.vanilla_model.y_input: y})

      bit1 = sess.run([self.model1.y_pred], feed_dict={self.model1.x_input: x1,
                                                                 self.model1.y_input: y})
      local1 = sess.run([self.model3.y_pred], feed_dict={self.model3.x_input: x3,
                                                                   self.model3.y_input: y})

      sq1_acc = 1 - np.sum(y_cur1 == self.Y) / (float(len(self.Y)))
      sq2_acc = 1 - np.sum(y_cur2 == self.Y) / (float(len(self.Y)))
      sq3_acc = 1 - np.sum(y_cur3 == self.Y) / (float(len(self.Y)))
      van_acc = 1 - np.sum(y_van == self.Y) / (float(len(self.Y)))
      min_acc = min(sq1_acc, sq2_acc, sq3_acc)


      # Let's first check if the models are outputting things correctly


      y_rc1 = np.argmax(self.rc1.predict(x_van), axis=1)
      y_rc2 = np.argmax(self.rc2.predict(x_van), axis=1)
      y_rc3 = np.argmax(self.rc3.predict(x_van), axis=1)

      sq1_acc_rc = 1 - np.sum(y_rc1 == self.Y) / (float(len(self.Y)))
      sq2_acc_rc = 1 - np.sum(y_rc2 == self.Y) / (float(len(self.Y)))
      sq3_acc_rc = 1 - np.sum(y_rc3 == self.Y) / (float(len(self.Y)))

      print("Itr: ", i, " Loss: ", l, " Reg Loss: ", r_loss)
      print(" With MY Classifiers     ==  Bit Depth: ", sq1_acc, " Median Depth: ", sq2_acc, " Non local means:", sq3_acc, " Vanilla :", van_acc)
      print(" With Robust Classifiers ==  Bit Depth: ", sq1_acc_rc, " Median Depth: ", sq2_acc_rc, " Non local means:", sq3_acc_rc )

      if np.array_equal(van_bit, van_local) == True and np.array_equal(y_rc1, y_rc3) == False:
        print (" ALERT Vanilla different squeezers are the same ")
      if np.array_equal(y_cur1, y_cur3) == True and np.array_equal(y_cur1, y_van):
        print (" ALERT MY CLASSIFIER has gone NUTs, Bit depth, non_local and vanilla are the same")

      if np.array_equal(van_bit, y_cur1) == False:
        print(" Error! Vanilla and My Classifier Differ for Bit Depth ")
      if np.array_equal(van_local,y_cur3) == False:
        print(" Error! Vanilla and My Classifier Differ for Non Local Means")

      if np.array_equal(van_bit, y_rc1) == False:
        print(" Error!  Robust Classifier and Vanilla Differ for Bit Depth ")
      if np.array_equal(van_local,y_rc3) == False:
        print(" Error! Robust Classifier and Vanilla and Differ for Non Local Means")

      if np.array_equal(y_van, van_bit ) == True:
        print("Error! HOLY FAAAK : Bit Depth and Vanilla are same ")
      if np.array_equal(y_van, van_local) == True:
        print(" Error! HOLY FAAAK  : Non Local Squeezer and Vanilla are same ")

      if np.array_equal(y_cur1, bit1) == False:
        print("Error! HOLY FAAAK  2: Bit Depth Squeezer Behaves weirdly ")
      if np.array_equal(y_cur3, local1) == False:
        print(" Error! HOLY FAAAK  2: Non Local Squeezer Behaves Weiredly ")



      if np.array_equal(y_rc1, y_cur1) == False:
        print(" OMG !!! :o !!!! Bit Depth Mismatch")
      if np.array_equal(y_rc2, y_cur2) == False:
        print(" OMG !!! :o !!!! Median Mismatch")
      if np.array_equal(y_rc3, y_cur3) == False:
        print(" OMG !!! :o !!!! Non Local Mismatch ")

      if min_acc > 0.90 and r_loss < r_min:
        r_min = r_loss
        x_max = np.copy(x)
        sel = i

      x += self.a * np.sign(grad)
      x = np.clip(x, 0, 1)  # ensure valid pixel range
      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)

    print(" Selected i:", sel, " Reg Loss:", r_min)
    # x_max = np.clip(x_max, x_nat - self.epsilon, x_nat + self.epsilon)
    return x_max

class CombinedLinfPGDAttackCIFAR10TWO:
  def __init__(self, model_vanilla, model1, model2, model3, epsilon, k, a, random_start, loss_func, Y,
               sq1, sq2, sq3):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point.
       This simulatenously runs the Combined Linf attack for 4 models
       """
    self.vanilla_model = model_vanilla
    self.median_model = model2



    self.k = k
    self.a = a
    self.Y = np.argmax(Y, axis=1)  # Target Labels
    self.rand = random_start
    self.epsilon = epsilon

    self.sq_bit = sq1
    self.sq_local = sq3

    self.rc_bit  = FeatureSqueezingRC(self.vanilla_model.keras_model, "FeatureSqueezing?squeezer=bit_depth_5")
    self.rc_median = FeatureSqueezingRC(self.vanilla_model.keras_model, "FeatureSqueezing?squeezer=median_filter_2_2")
    self.rc_local = FeatureSqueezingRC(self.vanilla_model.keras_model,
                                  "FeatureSqueezing?squeezer=non_local_means_color_13_3_2")

    self.vanilla_y_softmax = tf.nn.softmax(self.vanilla_model.pre_softmax)
    self.median_y_softmax = tf.nn.softmax( self.median_model.pre_softmax)
    self.t = self.vanilla_y_softmax - self.median_y_softmax
    self.diff_softmax = tf.nn.relu(tf.reduce_sum(tf.multiply(self.t, self.t), axis=1) - 1.2)
    # Need to normalize this
    self.reg_loss = 100 * tf.reduce_sum(self.diff_softmax)


    if loss_func != 'xent':
      print('Unknown loss function. Defaulting to cross-entropy')

    self.loss = self.vanilla_model.xent + self.median_model.xent + self.reg_loss

    self.grad = tf.gradients(self.loss, self.median_model.x_input)[0] + \
                2 * tf.gradients(self.loss, self.vanilla_model.x_input)[0]

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
      # Performing BPDA
      x_van = reduce_precision_py(x, 256)
      x_sq_bit = self.sq_bit(x_van)
      x_sq_local = self.sq_local(x_van)


      if np.array_equal(x_van, x_sq_bit) == True or np.array_equal(x_sq_local, x_van) == True:
        print(" Problem with the squeezer")

      grad, l, y_vanilla, y_median, r_loss = sess.run([self.grad, self.loss, self.vanilla_model.y_pred,
                                                       self.median_model.y_pred, self.reg_loss],
                                                      feed_dict={self.vanilla_model.x_input: x_van,
                                                                 self.median_model.x_input:  x_van,
                                                                 self.vanilla_model.y_input: y,
                                                                 self.median_model.y_input: y})
      y_robust_median = np.argmax(self.rc_median.predict(x_van), axis=1)


      median_accuracy        = 1 - np.sum(y_median == self.Y) / (float(len(self.Y)))
      vanilla_accuracy       = 1 - np.sum(y_vanilla == self.Y) / (float(len(self.Y)))
      robust_median_accuracy = 1 - np.sum(y_robust_median == self.Y) / (float(len(self.Y)))

      print("======  Itr: ", i, " Loss: ", l, " Reg Loss: ", r_loss)
      tempilate = ('Vanilla Accuracy: ({:.3}%)   Median Accuracy: ({:.3f}%)  Robust Median Accuracy ({:.3f}%)')

      print(tempilate.format(vanilla_accuracy,  median_accuracy, robust_median_accuracy))

      print(" !!![ Total Disagreements ]:", np.sum(np.absolute(y_median - y_robust_median)))

      if np.array_equal(y_robust_median, y_median) == False:
        print(" [ERROR] Median Model predictions differ from Robust Classifier Median")

      min_acc = min(median_accuracy ,vanilla_accuracy)
      if min_acc > max_acc:
        max_acc = min_acc
        sel = i
        x_max = np.copy(x)
        r_min = r_loss

      if min_acc >= max_acc and r_loss < r_min:
        self.a = 0.001
        r_min = r_loss
        x_max = np.copy(x)
        sel = i

      x += self.a * np.sign(grad)
      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 1)  # ensure valid pixel range

    print(" Selected i:", sel, " Reg Loss:", r_min)
    # x_max = np.clip(x_max, x_nat - self.epsilon, x_nat + self.epsilon)
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

class CombinedLinfPGDAttackCIFAR10:
  def __init__(self, model_vanilla, model1, model2, model3, epsilon, k, a, random_start, loss_func, Y,
               sq1, sq2, sq3):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point.
       This simulatenously runs the Combined Linf attack for 4 models
       """
    self.vanilla_model = model_vanilla
    self.median_model = model2

    self.LAMBDA = FLAGS.reg_lambda_x
    print(" ----------- ^^^^^^^^^ ^_^ ^^^^^^^^ ------------- REG LAMBDA X:  ", self.LAMBDA)
    self.k = k
    self.a = a
    self.Y = np.argmax(Y, axis=1)  # Target Labels
    self.rand = random_start
    self.epsilon = epsilon

    self.sq_bit = sq1
    self.sq_local = sq3

    self.cur_x = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, 3))

    self.rc_bit  = FeatureSqueezingRC(self.vanilla_model.keras_model, "FeatureSqueezing?squeezer=" + FLAGS.bit_depth_filter)
    self.rc_median = FeatureSqueezingRC(self.vanilla_model.keras_model, "FeatureSqueezing?squeezer="+ FLAGS.median_filter)
    self.rc_local = FeatureSqueezingRC(self.vanilla_model.keras_model,
                                  "FeatureSqueezing?squeezer="+ FLAGS.non_local_filter)

    self.x_diff_vanilla = self.cur_x - self.vanilla_model.x_input
    self.x_reg_loss_vanilla = self.LAMBDA * tf.reduce_sum(tf.multiply(self.x_diff_vanilla, self.x_diff_vanilla))

    self.x_diff_median = self.cur_x - self.median_model.x_input
    self.x_reg_loss_median = self.LAMBDA * tf.reduce_sum(tf.multiply(self.x_diff_median, self.x_diff_median))

    if loss_func != 'xent':
      print('Unknown loss function. Defaulting to cross-entropy')
    self.loss_vanilla = self.vanilla_model.xent + self.x_reg_loss_vanilla
    self.loss_median = self.median_model.xent + self.x_reg_loss_median

    self.grad_vanilla = tf.gradients(self.loss_vanilla, self.vanilla_model.x_input)[0]
    self.grad_median = tf.gradients(self.loss_median, self.median_model.x_input)[0]

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

    bit_depth_sq = get_squeezer_by_name(FLAGS.bit_depth_filter, "python")
    non_local_sq = get_squeezer_by_name(FLAGS.non_local_filter, "python")

    for i in range(self.k):
      x_van = reduce_precision_py(x, 256)
      x_bit = self.sq_bit(x_van)
      x_local = self.sq_local(x_van)

      vanilla_grad, vanilla_loss, y_vanilla = sess.run([self.grad_vanilla, self.loss_vanilla,
                                                        self.vanilla_model.y_pred],
                                                        feed_dict={self.vanilla_model.x_input: x_van,
                                                                  self.vanilla_model.y_input: y,
                                                                   self.cur_x : x_nat})

      median_grad, median_loss, y_median = sess.run([self.grad_median, self.loss_median,
                                                     self.median_model.y_pred],
                                                    feed_dict={self.median_model.x_input: x_van,
                                                               self.median_model.y_input: y, self.cur_x : x_nat})
      bit_grad, bit_loss, y_bit = sess.run([self.grad_vanilla, self.loss_vanilla,
                                            self.vanilla_model.y_pred],
                                           feed_dict={self.vanilla_model.x_input: x_bit,
                                                      self.vanilla_model.y_input: y, self.cur_x : x_nat})

      local_grad, local_loss, y_local = sess.run([self.grad_vanilla, self.loss_vanilla,
                                                  self.vanilla_model.y_pred],
                                                 feed_dict={self.vanilla_model.x_input: x_local,
                                                            self.vanilla_model.y_input: y, self.cur_x : x_nat})

      y_robust_median = np.argmax(self.rc_median.predict(x_van), axis=1)
      y_robust_bit = np.argmax(self.rc_bit.predict(x_van), axis=1)
      y_robust_local = np.argmax(self.rc_local.predict(x_van), axis = 1)



      vanilla_acc = 1.0 - np.sum(y_vanilla == self.Y) / (float(len(self.Y)))
      median_acc = 1.0 - np.sum(y_median == self.Y) / (float(len(self.Y)))
      bit_acc = 1.0 - np.sum(y_bit == self.Y) / (float(len(self.Y)))
      local_acc = 1.0 - np.sum(y_local == self.Y) / (float(len(self.Y)))

      min_acc = min(vanilla_acc, median_acc, bit_acc, local_acc)
      if (min_acc > max_acc):
        max_acc = min_acc
        x_max = np.copy(x)
        sel = i

      if (i > 15):
          self.a = 0.01

      print(" Iteration : ", i)
      tempilate = ('Vanilla: ({:.3f}%)   Median: ({:.3f}%)  Bit: ({:.3f}%) Non-local: ({:.3f}%)')
      print(tempilate.format(vanilla_acc, median_acc, bit_acc, local_acc))

      if not np.array_equal(y_robust_median, y_median):
          print("Medians Disagree")
      if not np.array_equal(y_robust_bit, y_bit):
          print("Bit Depth Disagree")
      if not np.array_equal(y_robust_local, y_local):
          print("Local Disagree")

      grad = vanilla_grad + median_grad + bit_grad + local_grad
      x += self.a * np.sign(grad)
      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 1)

    print("Selected Iteration:", sel)

    return x_max

class CombinedLinfPGDAttackDEBUG:
  def __init__(self, model_vanilla, model1, model2, model3, epsilon, k, a, random_start, loss_func, Y,
               sq1, sq2, sq3):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point.
       This simulatenously runs the Combined Linf attack for 4 models
       """
    self.vanilla_model = model3
    self.median_model = model2
    self.bit_model = model1

    self.LAMBDA = FLAGS.reg_lambda_x
    print(" ----------- ^^^^^^^^^ ^_^ ^^^^^^^^ ------------- REG LAMBDA X:  ", self.LAMBDA)
    print(" ----------- ^^^^^^^^^ ^_^ ^^^^^^^^ ------------- TOTAL REG LOSS:  ", FLAGS.tot_reg_loss)
    self.k = k
    self.a = a
    self.Y = np.argmax(Y, axis=1)  # Target Labels
    self.rand = random_start
    self.epsilon = epsilon

    self.sq_bit = sq1
    self.sq_local = sq3


    self.cur_x = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, 3))
    self.rc_bit  = FeatureSqueezingRC(self.vanilla_model.keras_model, "FeatureSqueezing?squeezer=" + FLAGS.bit_depth_filter)
    self.rc_median = FeatureSqueezingRC(self.vanilla_model.keras_model, "FeatureSqueezing?squeezer="+ FLAGS.median_filter)
    self.rc_local = FeatureSqueezingRC(self.vanilla_model.keras_model,
                                  "FeatureSqueezing?squeezer="+ FLAGS.non_local_filter)

    self.check = tf.reduce_sum(1 - tf.to_float(tf.equal(self.bit_model.x_input, self.vanilla_model.x_input)))

    self.x_diff_vanilla = self.cur_x - self.vanilla_model.x_input
    self.x_reg_loss_vanilla = self.LAMBDA * tf.reduce_sum(tf.multiply(self.x_diff_vanilla, self.x_diff_vanilla))

    self.x_diff_median = self.cur_x - self.median_model.x_input
    self.x_reg_loss_median = self.LAMBDA * tf.reduce_sum(tf.multiply(self.x_diff_median, self.x_diff_median))

    self.x_diff_bit = self.cur_x - self.bit_model.x_input
    self.x_reg_loss_bit = self.LAMBDA * tf.reduce_sum(tf.multiply(self.x_diff_bit, self.x_diff_bit))

    if loss_func != 'xent':
      print('Unknown loss function. Defaulting to cross-entropy')

    # (TODO:) Change this to be the difference between bit and vanilla

    THRESHOLD = 1.2
    self.vanilla_y_softmax = tf.nn.softmax(self.vanilla_model.pre_softmax)
    self.bit_y_softmax = tf.nn.softmax(self.bit_model.pre_softmax)
    self.t = self.vanilla_y_softmax - self.bit_y_softmax
    self.diff_softmax = tf.nn.relu(tf.reduce_sum(tf.multiply(self.t, self.t), axis=1) - THRESHOLD)
    self.reg_loss = FLAGS.tot_reg_loss * tf.reduce_sum(self.diff_softmax)

    self.loss_vanilla = self.vanilla_model.xent + self.x_reg_loss_vanilla
    self.loss_median = self.median_model.xent + self.x_reg_loss_median
    self.loss_bit = self.bit_model.xent + 10 * self.x_reg_loss_bit

    self.grad_vanilla = tf.gradients(self.loss_vanilla, self.vanilla_model.x_input)[0]
    self.grad_median = tf.gradients(self.loss_median, self.median_model.x_input)[0]
    self.grad_bit = tf.gradients(self.loss_bit, self.bit_model.x_input)[0]

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

    bit_depth_sq = get_squeezer_by_name(FLAGS.bit_depth_filter, "python")
    non_local_sq = get_squeezer_by_name(FLAGS.non_local_filter, "python")

    for i in range(self.k):
      x_van = reduce_precision_py(x, 256)
      x_bit = self.sq_bit(x_van)
      x_local = self.sq_local(x_van)

      vanilla_grad, vanilla_loss, y_vanilla, r_loss, \
      check, bit_grad1, bit_loss1, y_bit1 = \
                  sess.run([self.grad_vanilla, self.loss_vanilla,self.vanilla_model.y_pred,
                            self.reg_loss, self.check, self.grad_bit, self.loss_bit,self.bit_model.y_pred],
                           feed_dict={
                              self.vanilla_model.x_input: x_van,
                              self.vanilla_model.y_input: y,
                              self.cur_x: x_nat,
                              self.bit_model.x_input: x_bit,
                              self.bit_model.y_input: y })

      print("  =========================== Iteration =========================== : ", i)
      print("******** Check Value:   ", check)
      if np.array_equal(x_van, x_bit):
          print("!!!!!!!!!!!!!!!!!!!!!!!!!! X_vanilla and X_bit are the same")
      if check == 0 and not np.array_equal(x_van, x_bit):
        print (" The same value is being passed to both vanilla and bit models")

      bit_grad, bit_loss, y_bit = sess.run([self.grad_bit, self.loss_bit,self.bit_model.y_pred],
                                           feed_dict={ self.cur_x: x_nat,
                                                       self.bit_model.x_input: x_bit,
                                                        self.bit_model.y_input: y })

      if not np.array_equal(y_bit, y_bit1):
          print(" -----------------------------------------------------------------")
          print(" The two Bit predictions differ")
          print(" From the combined    Model :", y_bit1)
          print(" From the standalone  Model :", y_bit)
          print(" From the vanilla Model: ", y_vanilla)
          print(" -----------------------------------------------------------------")

      median_grad, median_loss, y_median = sess.run([ self.grad_median, self.loss_median,
                                                      self.median_model.y_pred ],
                                          feed_dict = {self.cur_x: x_nat,
                                                       self.median_model.x_input: x_van,
                                                      self.median_model.y_input: y })

      local_grad, local_loss, y_local = sess.run([self.grad_vanilla, self.loss_vanilla,
                                                  self.vanilla_model.y_pred],
                                                 feed_dict={self.vanilla_model.x_input: x_local,
                                                            self.vanilla_model.y_input: y, self.cur_x : x_nat})

      y_robust_median = np.argmax(self.rc_median.predict(x_van), axis=1)
      y_robust_bit = np.argmax(self.rc_bit.predict(x_van), axis=1)
      y_robust_local = np.argmax(self.rc_local.predict(x_van), axis = 1)



      vanilla_acc = 1.0 - np.sum(y_vanilla == self.Y) / (float(len(self.Y)))
      median_acc = 1.0 - np.sum(y_median == self.Y) / (float(len(self.Y)))
      bit_acc = 1.0 - np.sum(y_bit == self.Y) / (float(len(self.Y)))
      local_acc = 1.0 - np.sum(y_local == self.Y) / (float(len(self.Y)))

      min_acc = min(vanilla_acc, median_acc, bit_acc, local_acc)
      if (min_acc > max_acc):
        max_acc = min_acc
        x_max = np.copy(x)
        sel = i

      if (i > 15):
          self.a = 0.01


      tempilate = ('Vanilla: ({:.3f}%)   Median: ({:.3f}%)  Bit: ({:.3f}%) Non-local: ({:.3f}%) Reg Loss : ({:.3f}) ')
      print(tempilate.format(vanilla_acc, median_acc, bit_acc, local_acc, r_loss))

      if not np.array_equal(y_robust_median, y_median):
          print("!!!!!!!! [ERROR!! --- M ----] Medians Disagree")
      if not np.array_equal(y_robust_bit, y_bit):
          print("!!!!!!!! [ERROR!! ---- B ----]Bit Depth Disagree")
      if not np.array_equal(y_robust_local, y_local):
          print("!!!!!!!! [ERROR!!------- L ------ ] Local Disagree")

      grad = vanilla_grad + median_grad + bit_grad + local_grad
      x += self.a * np.sign(grad)
      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 1)

    print("Selected Iteration:", sel)

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
