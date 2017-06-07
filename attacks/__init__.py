from future.standard_library import install_aliases
install_aliases()
from urllib import parse as urlparse

import pickle
import numpy as np
import os
import time

from .cleverhans_wrapper import generate_fgsm_examples, generate_jsma_examples, generate_bim_examples
from .carlini_wrapper import generate_carlini_l2_examples, generate_carlini_li_examples, generate_carlini_l0_examples

def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False


def get_next_class(Y_test):
    num_classes = Y_test.shape[1]
    Y_test_labels = np.argmax(Y_test, axis=1)
    Y_test_labels = (Y_test_labels + 1) % num_classes
    return np.eye(num_classes)[Y_test_labels]

# TODO: replace pickle with .h5
def maybe_generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params, use_cache = False):
    x_adv_fpath = use_cache
    if use_cache and os.path.isfile(x_adv_fpath):
        X_adv, duration = pickle.load(open(x_adv_fpath, "rb"))
    else:
        time_start = time.time()
        X_adv = generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params)
        duration = time.time() - time_start

        if use_cache:
            pickle.dump((X_adv, duration), open(x_adv_fpath, 'wb'))
    return X_adv, duration


def parse_attack_string(attack_string):
    if '?' in attack_string:
        attack_name, attack_params = attack_string.split('?')
    else:
        attack_name, attack_params = attack_string, ''
    attack_name = attack_name.lower()
    attack_params = urlparse.parse_qs(attack_params)
    attack_params = dict( (k, v if len(v)>1 else v[0] ) for k,v in attack_params.items())

    for k,v in attack_params.items():
        if k in ['batch_size', 'max_iterations']:
            attack_params[k] = int(v)
        elif v == 'true':
            attack_params[k] = True
        elif v == 'false':
            attack_params[k] = False
        elif isfloat(v):
            attack_params[k] = float(v)
    return attack_name, attack_params

def generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params):
    batch_size = 100
    if attack_name == 'fgsm':
        X_adv = generate_fgsm_examples(sess, model, x, X, batch_size, attack_params)
    elif attack_name == 'jsma':
        X_adv = generate_jsma_examples(sess, model, x, y, X, Y, attack_params)
    elif attack_name == 'bim':
        X_adv = generate_bim_examples(sess, model, x, y, X, Y, attack_params)
    elif attack_name == 'carlinil2':
        X_adv = generate_carlini_l2_examples(sess, model, x, y, X, Y, attack_params)
    elif attack_name == 'carlinili':
        X_adv = generate_carlini_li_examples(sess, model, x, y, X, Y, attack_params)
    elif attack_name == 'carlinil0':
        X_adv = generate_carlini_l0_examples(sess, model, x, y, X, Y, attack_params)
        

    return X_adv
