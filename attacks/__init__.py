from future.standard_library import install_aliases
install_aliases()
from urllib import parse as urlparse

import pickle
import numpy as np
import os
import time

from .cleverhans_wrapper import generate_fgsm_examples, generate_jsma_examples, generate_bim_examples
from .carlini_wrapper import generate_carlini_l2_examples, generate_carlini_li_examples, generate_carlini_l0_examples
from .deepfool_wrapper import generate_deepfool_examples, generate_universal_perturbation_examples
from .adaptive.adaptive_adversary import generate_adaptive_carlini_l2_examples
from .pgd.pgd_wrapper import generate_pgdli_examples, bpda_generate_pgdli_examples


# TODO: replace pickle with .h5 for Python 2/3 compatibility issue.
def maybe_generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params,
                                use_cache=False, verbose=True, attack_log_fpath=None):
    x_adv_fpath = use_cache
    if use_cache and os.path.isfile(x_adv_fpath):
        print ("Loading adversarial examples from [%s]." % os.path.basename(x_adv_fpath))
        X_adv, duration = pickle.load(open(x_adv_fpath, "rb"))
    else:
        time_start = time.time()
        X_adv = generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params,
                                      verbose, attack_log_fpath)
        duration = time.time() - time_start

        if not isinstance(X_adv, np.ndarray):
            X_adv, aux_info = X_adv
        else:
            aux_info = {}

        aux_info['duration'] = duration

        if use_cache:
            pickle.dump((X_adv, aux_info), open(x_adv_fpath, 'wb'))
    return X_adv, duration


"""
def maybe_combined_generate_pgdli_examples(sess, vanilla_model, model1, model2, model3,  x, y, x_bit, x_local, x_median, X, Y, attack_params,
                                           use_cache=False, verbose=True, attack_log_fpath=None, sq1=lambda x:x,
                                           sq2=lambda x:x, sq3=lambda x:x ):
    x_adv_fpath = use_cache
    if use_cache and os.path.isfile(x_adv_fpath):
        print ("Loading adversarial examples from [%s]." % os.path.basename(x_adv_fpath))
        X_adv, duration = pickle.load(open(x_adv_fpath, "rb"))
    else:
        time_start = time.time()
        X_adv = combined_generate_pgdli_examples(sess, vanilla_model, model1, model2, model3,  x, y, x_bit, x_local, x_median, X, Y,
                                                 attack_params, sq1, sq2, sq3)
        duration = time.time() - time_start

        if not isinstance(X_adv, np.ndarray):
            X_adv, aux_info = X_adv
        else:
            aux_info = {}

        aux_info['duration'] = duration

        if use_cache:
            pickle.dump((X_adv, aux_info), open(x_adv_fpath, 'wb'))
    return X_adv, duration
"""


def maybe_median_generate_adv_examples(sess, model, vanilla_model, x, y, X, Y, attack_params,
                                           use_cache=False, verbose=True, attack_log_fpath=None):
    x_adv_fpath = use_cache
    if use_cache and os.path.isfile(x_adv_fpath):
        print ("Loading adversarial examples from [%s]." % os.path.basename(x_adv_fpath))
        X_adv, duration = pickle.load(open(x_adv_fpath, "rb"))
    else:
        time_start = time.time()
        X_adv = generate_pgdli_examples(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath,
                                                 vanilla_model)
        duration = time.time() - time_start

        if not isinstance(X_adv, np.ndarray):
            X_adv, aux_info = X_adv
        else:
            aux_info = {}

        aux_info['duration'] = duration

        if use_cache:
            pickle.dump((X_adv, aux_info), open(x_adv_fpath, 'wb'))
    return X_adv, duration


def maybe_bpda_generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params, use_cache=False, verbose=True,
                               attack_log_fpath=None, squeezer=lambda x:x):
    x_adv_fpath = use_cache
    if use_cache and os.path.isfile(x_adv_fpath):
        print ("Loading adversarial examples from [%s]." % os.path.basename(x_adv_fpath))
        X_adv, duration = pickle.load(open(x_adv_fpath, "rb"))
    else:
        time_start = time.time()
        X_adv = generate_pgdli_examples
        duration = time.time() - time_start

        if not isinstance(X_adv, np.ndarray):
            X_adv, aux_info = X_adv
        else:
            aux_info = {}

        aux_info['duration'] = duration

        if use_cache:
            pickle.dump((X_adv, aux_info), open(x_adv_fpath, 'wb'))
    return X_adv, duration

def bpda_generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params, verbose, attack_log_fpath,
                          squeezer=lambda x:x):
    # Only support PGD for now
    if attack_name != 'pgdli':
        raise NotImplementedError("Unsuported attack [%s]." % attack_name)

    X_adv = bpda_generate_pgdli_examples(sess, model, x, y, X, Y, attack_params,
                                         verbose, attack_log_fpath, squeezer)
    return X_adv

def generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params, verbose, attack_log_fpath):
    if attack_name == 'none':
        return X
    elif attack_name == 'fgsm':
        generate_adv_examples_func = generate_fgsm_examples
    elif attack_name == 'jsma':
        generate_adv_examples_func = generate_jsma_examples
    elif attack_name == 'bim':
        generate_adv_examples_func = generate_bim_examples
    elif attack_name == 'carlinil2':
        generate_adv_examples_func = generate_carlini_l2_examples
    elif attack_name == 'carlinili':
        generate_adv_examples_func = generate_carlini_li_examples
    elif attack_name == 'carlinil0':
        generate_adv_examples_func = generate_carlini_l0_examples
    elif attack_name == 'deepfool':
        generate_adv_examples_func = generate_deepfool_examples
    elif attack_name == 'unipert':
        generate_adv_examples_func = generate_universal_perturbation_examples
    elif attack_name == 'adaptive_carlini_l2':
        generate_adv_examples_func = generate_adaptive_carlini_l2_examples
    elif attack_name == 'pgdli':
        generate_adv_examples_func = generate_pgdli_examples
    else:
        raise NotImplementedError("Unsuported attack [%s]." % attack_name)

    X_adv = generate_adv_examples_func(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath)

    return X_adv

