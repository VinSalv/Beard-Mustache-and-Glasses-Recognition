import argparse
import csv

import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from utils.utils import Feature, get_labels_from


def get_ground_truth_path(config):
    return config['dataset']['folder']


def get_predictions_path(config):
    return config['output']['results']


def get_ground_truth_file(config):
    return config['labels']['test']


def get_predictions_file(config):
    return config['labels']['results']


def get_features_summary(labels):
    results = {Feature.BEARD.value: [], Feature.MUSTACHE.value: [], Feature.GLASSES.value: []}
    for image in labels.keys():
        results[Feature.BEARD.value].append(labels[image][Feature.BEARD.value])
        results[Feature.MUSTACHE.value].append(labels[image][Feature.MUSTACHE.value])
        results[Feature.GLASSES.value].append(labels[image][Feature.GLASSES.value])

    return results


def get_scores(ground_truth, predictions):
    accuracy = accuracy_score(ground_truth, predictions)
    balanced_accuracy = balanced_accuracy_score(ground_truth, predictions)
    return accuracy, balanced_accuracy


def init_parameter():
    parser = argparse.ArgumentParser(description='Final Project evaluation')
    parser.add_argument("--gt_path",
                        type=str,
                        default=None,
                        help="Fullpath to File CSV with groundtruth\n "
                             "Example: ./foo/foo.csv")
    parser.add_argument("--res_path",
                        type=str,
                        default=None,
                        help="Fullpath to File CSV with prediction results\n "
                             "Example: ./foo/foo.csv")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    with open('config.yml') as file:
        config = yaml.full_load(file)

    args = init_parameter()

    # Init params
    ground_truth_path = get_ground_truth_path(config)
    predictions_path = get_predictions_path(config)
    ground_truth_file = get_ground_truth_file(config)
    predictions_file = get_predictions_file(config)

    ground_truth_filepath = args.gt_path or (ground_truth_path + ground_truth_file)
    predictions = args.res_path or (predictions_path + predictions_file)

    test_label = get_labels_from(csv, ground_truth_filepath)
    predict_label = get_labels_from(csv, predictions)

    ground_truth = get_features_summary(test_label)
    predictions_results = get_features_summary(predict_label)

    # Performance metrics
    beard_accuracy, beard_balanced_accuracy = get_scores(ground_truth[Feature.BEARD.value],
                                                         predictions_results[Feature.BEARD.value])
    confusion_matrix_beard = confusion_matrix(ground_truth[Feature.BEARD.value],
                                              predictions_results[Feature.BEARD.value])
    moustache_accuracy, moustache_balanced_accuracy = get_scores(ground_truth[Feature.MUSTACHE.value],
                                                                 predictions_results[Feature.MUSTACHE.value])
    confusion_matrix_mustache = confusion_matrix(ground_truth[Feature.MUSTACHE.value],
                                                 predictions_results[Feature.MUSTACHE.value])
    glasses_accuracy, glasses_balanced_accuracy = get_scores(ground_truth[Feature.GLASSES.value],
                                                             predictions_results[Feature.GLASSES.value])
    confusion_matrix_glasses = confusion_matrix(ground_truth[Feature.GLASSES.value],
                                                predictions_results[Feature.GLASSES.value])

    avg_accuracy = (beard_accuracy + moustache_accuracy + glasses_accuracy) / 3
    avg_balanced_accuracy = (beard_balanced_accuracy + moustache_balanced_accuracy + glasses_balanced_accuracy) / 3
    fas = avg_accuracy + avg_balanced_accuracy

    print("\nbeard_accuracy: %.3f\n"
          "beard_balanced_accuracy: %.3f\n"
          f"confusion_matrix_beard: \n{confusion_matrix_beard}\n\n"
          "moustache_accuracy: %.3f\n"
          "moustache_balanced_accuracy: %.3f\n"
          f"confusion_matrix_mustache: \n{confusion_matrix_mustache}\n\n"
          "glasses_accuracy: %.3f\n"
          "glasses_balanced_accuracy: %.3f\n"
          f"confusion_matrix_glasses: \n{confusion_matrix_glasses}\n\n"
          "avg_accuracy: %.3f\n"
          "avg_balanced_accuracy: %.3f\n\n"
          "fas: %.3f" % (beard_accuracy,
                         beard_balanced_accuracy,
                         moustache_accuracy,
                         moustache_balanced_accuracy,
                         glasses_accuracy,
                         glasses_balanced_accuracy,
                         avg_accuracy,
                         avg_balanced_accuracy,
                         fas))
