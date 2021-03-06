"""
    EOT Attack for all the filters that they consider for CIFAR and ImageNet, which are:
    Bit_depth 4, 5
    Median, 2, 3
    Non local means: 11_3_4, 11_3_2, 13_3_4, 13_3_2
    Flags are exactly the same as before.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from scipy import ndimage

import os
import pdb
import pickle
import time
import random

import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from utils.squeeze import get_squeezer_by_name
from attacks.pgd.pgd_wrapper import eot_adversarial_attack


FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_name', 'MNIST', 'Supported: MNIST, CIFAR-10, ImageNet.')
flags.DEFINE_string('model_name', 'cleverhans', 'Supported: cleverhans, cleverhans_adv_trained and carlini for MNIST; carlini and DenseNet for CIFAR-10;  ResNet50, VGG19, Inceptionv3 and MobileNet for ImageNet.')

flags.DEFINE_boolean('select', True, 'Select correctly classified examples for the experiement.')
flags.DEFINE_integer('nb_examples', 200, 'The number of examples selected for attacks.')
flags.DEFINE_boolean('balance_sampling', False, 'Select the same number of examples for each class.')
flags.DEFINE_boolean('test_mode', False, 'Only select one sample for each class.')

flags.DEFINE_string('attacks', "FGSM?eps=0.1;BIM?eps=0.1&eps_iter=0.02;JSMA?targeted=next;CarliniL2?targeted=next&batch_size=100&max_iterations=1000;CarliniL2?targeted=next&batch_size=100&max_iterations=1000&confidence=2", 'Attack name and parameters in URL style, separated by semicolon.')
flags.DEFINE_float('clip', -1, 'L-infinity clip on the adversarial perturbations.')
flags.DEFINE_string('robustness', '', 'Supported: FeatureSqueezing.')

flags.DEFINE_string('result_folder', "results", 'The output folder for results.')
flags.DEFINE_boolean('verbose', False, 'Stdout level. The hidden content will be saved to log files anyway.')

flags.DEFINE_string('detection', '', 'Supported: feature_squeezing.')
flags.DEFINE_boolean('detection_train_test_mode', True, 'Split into train/test datasets.')
flags.DEFINE_boolean('visualize', True, 'Output the image examples for each attack, enabled by default.')
flags.DEFINE_float('reg_x', 0.1, "Regularizer for x to ensure x_adv is not too different from x")
flags.DEFINE_float('reg_lambda', 100.0, "Coefficient for Regularization loss")
flags.DEFINE_float('threshold', 1.0, " Threshold for the Filters")
flags.DEFINE_integer('image_size', 32, ' Image size. CIFAR Image Size by Default')
flags.DEFINE_string('squeezers', "bit_depth_4;bit_depth_5;median_filter_2_2;median_filter_3_3;"
                                 "non_local_means_color_11_3_2;non_local_means_color_11_3_4;"
                                 "non_local_means_color_13_3_2;non_local_means_color_13_3_4",
                    " All the Squeezers used to generate the EOT attack ")
FLAGS.model_name = FLAGS.model_name.lower()

def load_tf_session():
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)
    print("Created TensorFlow session and set Keras backend.")
    return sess


def main(argv=None):
    # 0. Select a dataset.
    from datasets import MNISTDataset, CIFAR10Dataset, ImageNetDataset
    from datasets import get_correct_prediction_idx, evaluate_adversarial_examples, calculate_mean_confidence, \
        calculate_accuracy

    if FLAGS.dataset_name == "MNIST":
        dataset = MNISTDataset()
    elif FLAGS.dataset_name == "CIFAR-10":
        dataset = CIFAR10Dataset()
        FLAGS.image_size = 32           # Redundant for the Current Attack.
    elif FLAGS.dataset_name == "ImageNet":
        dataset = ImageNetDataset()
        FLAGS.image_size = 224          # Redundant for the Current Attack.

    # 1. Load a dataset.
    print("\n===Loading %s data..." % FLAGS.dataset_name)
    if FLAGS.dataset_name == 'ImageNet':
        if FLAGS.model_name == 'inceptionv3':
            img_size = 299
        else:
            img_size = 224
        X_test_all, Y_test_all = dataset.get_test_data(img_size, 0, 200)
    else:
        X_test_all, Y_test_all = dataset.get_test_dataset()

    # Randomized optimizations
    if FLAGS.dataset_name != "ImageNet":
        all_idx = np.arange(10000)
        np.random.shuffle(all_idx)
        selected_idx = all_idx[:(FLAGS.nb_examples * 2)]
        X_test_all, Y_test_all = X_test_all[selected_idx], Y_test_all[selected_idx]

    # 2. Load a trained model.
    sess = load_tf_session()
    keras.backend.set_learning_phase(0)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, dataset.image_size, dataset.image_size, dataset.num_channels))
    y = tf.placeholder(tf.float32, shape=(None, dataset.num_classes))

    sq_list = FLAGS.squeezers.split(';')
    print(" Squeezers used for EOT :", sq_list)

    x_s  = []
    squeezers = []
    models = []
    for squeezer in sq_list:
        x_s.append(
            tf.placeholder(tf.float32, shape=(None, dataset.image_size, dataset.image_size, dataset.num_channels)))
        if squeezer.startswith("median"):
            squeezers.append(lambda x: x)
            with tf.variable_scope(FLAGS.model_name + squeezer):
                cur_model = dataset.load_model_by_name(FLAGS.model_name, logits=False, input_range_type=1,
                                                           pre_filter=get_squeezer_by_name(
                                                               squeezer, 'tensorflow'))
                cur_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
                models.append(cur_model)
        else:
            squeezers.append(get_squeezer_by_name(squeezer, 'python'))
            with tf.variable_scope(FLAGS.model_name + "local" + squeezer):
                cur_model = dataset.load_model_by_name(FLAGS.model_name, logits=False, input_range_type=1)
                cur_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
                models.append(cur_model)


    with tf.variable_scope(FLAGS.model_name + "vanilla"):
        model_vanilla = dataset.load_model_by_name(FLAGS.model_name, logits=False, input_range_type=1)
        model_vanilla.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])

    # 3. Evaluate the trained model.
    # TODO: add top-5 accuracy for ImageNet.
    print ("Evaluating the pre-trained model...")

    # We use the Vanilla Model here for Prediction
    print("  ************************************************* Shape of X_test_all :", X_test_all.shape)
    Y_pred_all = model_vanilla.predict(X_test_all)
    mean_conf_all = calculate_mean_confidence(Y_pred_all, Y_test_all)
    accuracy_all = calculate_accuracy(Y_pred_all, Y_test_all)
    print('Test accuracy on raw legitimate examples %.4f' % (accuracy_all))
    print('Mean confidence on ground truth classes %.4f' % (mean_conf_all))


    # 4. Select some examples to attack.
    import hashlib
    from datasets import get_first_n_examples_id_each_class

    if FLAGS.select:
        # Filter out the misclassified examples.
        correct_idx = get_correct_prediction_idx(Y_pred_all, Y_test_all)
        if FLAGS.test_mode:
            # Only select the first example of each class.
            correct_and_selected_idx = get_first_n_examples_id_each_class(Y_test_all[correct_idx])
            selected_idx = [ correct_idx[i] for i in correct_and_selected_idx ]
        else:
            if not FLAGS.balance_sampling:
                selected_idx = correct_idx[:FLAGS.nb_examples]
            else:
                # select the same number of examples for each class label.
                nb_examples_per_class = int(FLAGS.nb_examples / Y_test_all.shape[1])
                correct_and_selected_idx = get_first_n_examples_id_each_class(Y_test_all[correct_idx], n=nb_examples_per_class)
                selected_idx = [ correct_idx[i] for i in correct_and_selected_idx ]
    else:
        selected_idx = np.array(range(FLAGS.nb_examples))

    from utils.output import format_number_range
    selected_example_idx_ranges = format_number_range(sorted(selected_idx))
    print ( "Selected %d examples." % len(selected_idx))
    print ( "Selected index in test set (sorted): %s" % selected_example_idx_ranges )
    X_test, Y_test, Y_pred = X_test_all[selected_idx], Y_test_all[selected_idx], Y_pred_all[selected_idx]

    # The accuracy should be 100%.
    accuracy_selected = calculate_accuracy(Y_pred, Y_test)
    mean_conf_selected = calculate_mean_confidence(Y_pred, Y_test)
    print('Test accuracy on selected legitimate examples %.4f' % (accuracy_selected))
    print('Mean confidence on ground truth classes, selected %.4f\n' % (mean_conf_selected))

    task = {}
    task['dataset_name'] = FLAGS.dataset_name
    task['model_name'] = FLAGS.model_name
    task['accuracy_test'] = accuracy_all
    task['mean_confidence_test'] = mean_conf_all

    task['test_set_selected_length'] = len(selected_idx)
    task['test_set_selected_idx_ranges'] = selected_example_idx_ranges
    task['test_set_selected_idx_hash'] = hashlib.sha1(str(selected_idx).encode('utf-8')).hexdigest()
    task['accuracy_test_selected'] = accuracy_selected
    task['mean_confidence_test_selected'] = mean_conf_selected

    task_id = "%s_%d_%s_%s" % \
            (task['dataset_name'], task['test_set_selected_length'], task['test_set_selected_idx_hash'][:5],
             task['model_name'], )

    FLAGS.result_folder = os.path.join(FLAGS.result_folder, task_id)
    if not os.path.isdir(FLAGS.result_folder):
        os.makedirs(FLAGS.result_folder)

    from utils.output import save_task_descriptor
    save_task_descriptor(FLAGS.result_folder, [task])


    # 5. Generate adversarial examples.

    from utils.squeeze import reduce_precision_py
    from utils.parameter_parser import parse_params
    attack_string_hash = hashlib.sha1(FLAGS.attacks.encode('utf-8')).hexdigest()[:5]
    sample_string_hash = task['test_set_selected_idx_hash'][:5]

    from datasets.datasets_utils import get_next_class, get_least_likely_class
    Y_test_target_next = get_next_class(Y_test)
    Y_test_target_ll = get_least_likely_class(Y_pred)

    X_test_adv_list = []
    X_test_adv_discretized_list = []
    Y_test_adv_discretized_pred_list = []

    attack_string_list = filter(lambda x:len(x)>0, FLAGS.attacks.lower().split(';'))
    to_csv = []

    X_adv_cache_folder = os.path.join(FLAGS.result_folder, 'adv_examples')
    adv_log_folder = os.path.join(FLAGS.result_folder, 'adv_logs')
    predictions_folder = os.path.join(FLAGS.result_folder, 'predictions')
    for folder in [X_adv_cache_folder, adv_log_folder, predictions_folder]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    predictions_fpath = os.path.join(predictions_folder, "legitimate.npy")
    np.save(predictions_fpath, Y_pred, allow_pickle=False)

    if FLAGS.clip >= 0:
        epsilon = FLAGS.clip
        print ("Clip the adversarial perturbations by +-%f" % epsilon)
        max_clip = np.clip(X_test + epsilon, 0, 1)
        min_clip = np.clip(X_test - epsilon, 0, 1)

    # NOTE : At the moment we only support single attacks and single detectors.
    for attack_string in attack_string_list:
        attack_name, attack_params = parse_params(attack_string)
        print ( "\nRunning attack: %s %s" % (attack_name, attack_params))

        if 'targeted' in attack_params:
            targeted = attack_params['targeted']
            print ("targeted value: %s" % targeted)
            if targeted == 'next':
                Y_test_target = Y_test_target_next
            elif targeted == 'll':
                Y_test_target = Y_test_target_ll
            elif targeted == False:
                attack_params['targeted'] = False
                Y_test_target = Y_test.copy()
        else:
            targeted = False
            attack_params['targeted'] = False
            Y_test_target = Y_test.copy()


        # Note that we use the attack model here instead of the vanilla model
        # Note that we pass in the Squeezer function for BPDA
        X_test_adv = eot_adversarial_attack(sess, model_vanilla, models, x, y, x_s,
                                            X_test, Y_test_target, attack_params, squeezers)


        if FLAGS.clip > 0:
            # This is L-inf clipping.
            X_test_adv = np.clip(X_test_adv, min_clip, max_clip)

        X_test_adv_list.append(X_test_adv)

        # 5.0 Output predictions.
        Y_test_adv_pred = model_vanilla.predict(X_test_adv)
        predictions_fpath = os.path.join(predictions_folder, "%s.npy" % attack_string)
        np.save(predictions_fpath, Y_test_adv_pred, allow_pickle=False)

        # 5.1 Evaluate the adversarial examples being discretized to uint8.
        print("\n---Attack (uint8): %s" % attack_string)
        # All data should be discretized to uint8.
        X_test_adv_discret = reduce_precision_py(X_test_adv, 256)
        X_test_adv_discretized_list.append(X_test_adv_discret)
        Y_test_adv_discret_pred = model_vanilla.predict(X_test_adv_discret)
        Y_test_adv_discretized_pred_list.append(Y_test_adv_discret_pred)

        # Y_test_adv_discret_pred is for the vanilla model
        rec = evaluate_adversarial_examples(X_test, Y_test, X_test_adv_discret, Y_test_target.copy(), targeted,
                                            Y_test_adv_discret_pred)
        rec['dataset_name'] = FLAGS.dataset_name
        rec['model_name'] = FLAGS.model_name
        rec['attack_string'] = attack_string
        rec['discretization'] = True
        to_csv.append(rec)

    from utils.output import write_to_csv
    attacks_evaluation_csv_fpath = os.path.join(FLAGS.result_folder,
                                                "%s_attacks_%s_evaluation.csv" % \
                                                (task_id, attack_string_hash))
    fieldnames = ['dataset_name', 'model_name', 'attack_string',  'discretization',
                  'success_rate', 'mean_confidence', 'mean_l2_dist', 'mean_li_dist', 'mean_l0_dist_value',
                  'mean_l0_dist_pixel']
    write_to_csv(to_csv, attacks_evaluation_csv_fpath, fieldnames)

    if FLAGS.visualize is True:
        from datasets.visualization import show_imgs_in_rows
        if FLAGS.test_mode or FLAGS.balance_sampling:
            selected_idx_vis = range(Y_test.shape[1])
        else:
            selected_idx_vis = get_first_n_examples_id_each_class(Y_test, 1)

        legitimate_examples = X_test[selected_idx_vis]

        rows = [legitimate_examples]
        rows += map(lambda x: x[selected_idx_vis], X_test_adv_list)

        img_fpath = os.path.join(FLAGS.result_folder, '%s_attacks_%s_examples.png' % (task_id, attack_string_hash))
        #show_imgs_in_rows(rows, img_fpath)
        print('\n===Adversarial image examples are saved in ', img_fpath)

        # TODO: output the prediction and confidence for each example, both legitimate and adversarial.

    # 6. Evaluate robust classification techniques.
    # Example: --robustness \
    #           "Base;FeatureSqueezing?squeezer=bit_depth_1;FeatureSqueezing?squeezer=median_filter_2;"
    if FLAGS.robustness != '':
        """
        Test the accuracy with robust classifiers.
        Evaluate the accuracy on all the legitimate examples.
        """
        from robustness import evaluate_robustness
        result_folder_robustness = os.path.join(FLAGS.result_folder, "robustness")
        fname_prefix = "robustness_summary"
        evaluate_robustness(FLAGS.robustness, model_vanilla, Y_test_all, X_test_all, Y_test, \
                            attack_string_list, X_test_adv_discretized_list,
                            fname_prefix, selected_idx_vis, result_folder_robustness)

    # 7. Detection experiment.
    # Example: --detection "FeatureSqueezing?distance_measure=l1&squeezers=median_smoothing_2,bit_depth_4,bilateral_filter_15_15_60;"
    if FLAGS.detection != '':
        from detections.base import DetectionEvaluator

        result_folder_detection = os.path.join(FLAGS.result_folder, "detection")
        csv_fname = "detection_summary.csv"
        de = DetectionEvaluator(model_vanilla, result_folder_detection, csv_fname, FLAGS.dataset_name)
        Y_test_all_pred = model_vanilla.predict(X_test_all)
        de.build_detection_dataset(X_test_all, Y_test_all, Y_test_all_pred, selected_idx,
                                   X_test_adv_discretized_list, Y_test_adv_discretized_pred_list,
                                   attack_string_list, attack_string_hash, FLAGS.clip,
                                   Y_test_target_next, Y_test_target_ll)
        de.evaluate_detections(FLAGS.detection)


if __name__ == '__main__':
    main()
