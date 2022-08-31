import argparse
import csv
import os
import shutil

import cv2 as cv
import dlib
import yaml

from utils.classifier.feature_classifier import FeatureClassifier
from utils.utils import Feature, extract_features_for_testing, get_labels_from


def get_features(config):
    return config['features']['with']


def get_predictor_path(config):
    return config['libs']['predictor']


def get_models_path(config):
    return config['models']


def get_image_path(config):
    return config['dataset']['test']


def get_test_csv_path(config):
    return config['dataset']['folder']


def get_output_path(config):
    return config['output']['results']


def get_output_file(config):
    return config['labels']['results']


def get_test_csv_file(config):
    return config['labels']['test']


def get_test_predictions(test_labels, img_test, features, path_models, detector, predictor):
    results = {}
    for image_name in test_labels.keys():
        img = cv.imread(os.path.join(img_test, image_name))

        results[image_name] = {}
        for feature in features:
            kind_of_feature = Feature(feature)
            try:
                X_val, y_val = extract_features_for_testing(img,
                                                            test_labels[image_name][feature],
                                                            detector,
                                                            predictor,
                                                            kind_of_feature)
                results[image_name][feature] = prediction(X_val,
                                                          path_models[feature],
                                                          kind_of_feature)
            except:
                results[image_name][feature] = [0]
    return results


def save_results(writer, predictions):
    print("\nWrite result.csv...")
    for image_name in predictions.keys():
        beard_value = predictions[image_name][Feature.BEARD.value].__getitem__(0)
        if beard_value == 1:
            mustache_value = 0
        else:
            mustache_value = predictions[image_name][Feature.MUSTACHE.value].__getitem__(0)
        glasses_value = predictions[image_name][Feature.GLASSES.value].__getitem__(0)
        writer.writerow([image_name, beard_value, mustache_value, glasses_value])


def prediction(X_val, path_model, kind_of_feature):
    model = FeatureClassifier()
    print("\nFetching " + str(kind_of_feature) + " model...")
    model.load(path_model)
    print("\nPrediction based on " + str(kind_of_feature) + " model...")
    y_pred = model.predict(X_val)
    return y_pred


def init_parameter():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--data",
                        type=str,
                        default=None,
                        help="Full path to Dataset labels\n "
                             "Example: ./foo/foo.csv")
    parser.add_argument("--images",
                        type=str,
                        default=None,
                        help="Path to Dataset folder\n "
                             "Example: ./foo/")
    parser.add_argument("--results",
                        type=str,
                        default=None,
                        help="Name of CSV file of the results\n "
                             "Example: results.csv")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    with open('config.yml') as file:
        config = yaml.full_load(file)

    args = init_parameter()

    # Init params
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(get_predictor_path(config))

    features = get_features(config)

    path_models = get_models_path(config)
    img_test = args.images or get_image_path(config)
    output_path = get_output_path(config)
    output_file = args.results or get_output_file(config)
    test_csv_path = get_test_csv_path(config)
    test_csv_file = get_test_csv_file(config)

    test_csv = args.data or (test_csv_path + test_csv_file)

    test_labels = get_labels_from(csv, test_csv)

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # Writing CSV results
    with open(output_path + output_file, 'w', newline='') as res_file:
        writer = csv.writer(res_file)
        predictions = get_test_predictions(test_labels,
                                           img_test,
                                           features,
                                           path_models,
                                           detector,
                                           predictor)
        save_results(writer, predictions)

    print("\nFinish!!")
