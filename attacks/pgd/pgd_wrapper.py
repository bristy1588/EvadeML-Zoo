
import warnings
from .pgd_attack import LinfPGDAttack, CombinedLinfPGDAttack, EOTLinfPGDAttack
from .pgd_attack import CombinedLinfPGDAttackDEBUG
from keras.models import Model
import tensorflow as tf
import numpy as np
from keras.utils import plot_model

from robustness.feature_squeezing import FeatureSqueezingRC

def override_params(default, update):
    for key in default:
        if key in update:
            val = update[key]
            if key == 'k':
                val = int(val)
            default[key] = val
            del update[key]

    if len(update) > 0:
        warnings.warn("Ignored arguments: %s" % update.keys())
    return default


class PGDModelWrapper:
    def __init__(self, keras_model, x, y, file_name="bristy"):
        model_logits = Model(inputs=keras_model.layers[0].input, outputs=keras_model.layers[-2].output)
        #plot_model(model_logits, to_file=file_name, show_shapes=True)
        self.x_input = x
        self.y_input = tf.argmax(y, 1)
        self.pre_softmax = model_logits(x)
        self.keras_model = keras_model
        #self.x_nat = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_input, logits=self.pre_softmax)
        self.xent = tf.reduce_sum(y_xent)
     # IMPORTANT !! Note that hard-coding this for van

        self.y_pred = tf.argmax(self.pre_softmax, 1)



def generate_pgdli_examples(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath, vanilla_model = None):
    model_for_pgd = PGDModelWrapper(model, x, y)
    if vanilla_model is None:
        vanilla_model = model
    vanilla_model_for_pgd = PGDModelWrapper(vanilla_model, x, y)
    params = {'model': model_for_pgd, 'epsilon': 0.3, 'k': 20, 'a':0.01, 'random_start':True,
                     'loss_func':'xent', 'squeezer' : lambda x:x, 'Y' : Y, 'vanilla_model' : vanilla_model_for_pgd}
    params = override_params(params, attack_params)
    attack = LinfPGDAttack(**params)

    Y_class = np.argmax(Y, 1)
    X_adv = attack.perturb(X, Y_class, sess)
    return X_adv

def bpda_generate_pgdli_examples(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath, squeezer=lambda x:x):
    model_for_pgd = PGDModelWrapper(model, x, y)
    params = {'model': model_for_pgd, 'epsilon': 0.3, 'k': 40, 'a':0.01, 'random_start':True,
                     'loss_func':'xent', 'squeezer' : squeezer, 'Y' : Y}
    params = override_params(params, attack_params)
    attack = LinfPGDAttack(**params)
    Y_class = np.argmax(Y, 1)
    X_adv = attack.perturb(X, Y_class, sess)
    return X_adv


def combined_generate_pgdli_examples(sess, model_vanilla,  model1, model2, model3,  x, y, x_bit, x_local, x_median, X, Y, attack_params, sq1,sq2,sq3):
    model_1_for_pgd = PGDModelWrapper(model1, x_bit, y)
    model_2_for_pgd = PGDModelWrapper(model2, x_local, y)
    model_3_for_pgd = PGDModelWrapper(model3, x_median, y)
    model_vanilla_for_pgd = PGDModelWrapper(model_vanilla, x, y)

    params = {'model1': model_1_for_pgd,  'model2': model_2_for_pgd,'model3': model_3_for_pgd, 'epsilon': 0.3,
              'k': 20, 'a': 0.01, 'random_start': True,'loss_func': 'xent', 'sq1': sq1, 'sq2': sq2, 'sq3': sq3, 'Y': Y,
              'model_vanilla': model_vanilla_for_pgd}

    params = override_params(params, attack_params)
    attack = CombinedLinfPGDAttackDEBUG(**params)
    Y_class = np.argmax(Y, 1)
    X_adv = attack.perturb(X, Y_class, sess)
    return X_adv

def combined_adversarial_attack(sess, model_vanilla, model_bit, model_median, model_local, x, y, x_bit,
                                x_local, x_median, X, Y, attack_params, sq_bit, sq_median,
                                sq_local ):
    print(" Entering Adversarial Attack ")
    model_vanilla_pgd = PGDModelWrapper(model_vanilla, x, y)
    model_bit_pgd = PGDModelWrapper(model_bit, x_bit, y)
    model_median_pgd = PGDModelWrapper(model_median, x_median, y)
    model_local_pgd = PGDModelWrapper(model_local, x_local, y)


    params = {'model_vanilla': model_vanilla_pgd,  'model_bit': model_bit_pgd, 'model_median': model_median_pgd,
              'model_local': model_local_pgd,  'epsilon': 0.3,'k': 20, 'a': 0.01, 'random_start': True,
              'loss_func': 'xent', 'sq_bit': sq_bit, 'sq_median': sq_median, 'sq_local': sq_local, 'Y': Y}

    params = override_params(params, attack_params)
    attack = CombinedLinfPGDAttack(**params)
    Y_class = np.argmax(Y, 1)
    X_adv = attack.perturb(X, Y_class, sess)
    return X_adv

def eot_adversarial_attack(sess, model_vanilla, models, x, y, x_s,  X, Y, attack_params, squeezers):
    other_models = []
    for (x_cur, model_cur) in zip(x_s, models):
        model_cur = PGDModelWrapper(model_cur, x_cur, y)
        other_models.append(model_cur)

    model_vanilla_pgd = PGDModelWrapper(model_vanilla, x, y)

    params = {'model_vanilla': model_vanilla_pgd, 'other_models': other_models, 'epsilon': 0.3, 'k': 20,
              'a': 0.01, 'random_start': True, 'loss_func': 'xent', 'squeezers': squeezers, 'Y': Y}

    params = override_params(params, attack_params)
    attack = EOTLinfPGDAttack(**params)
    Y_class = np.argmax(Y, 1)
    X_adv = attack.perturb(X, Y_class, sess)
    return X_adv