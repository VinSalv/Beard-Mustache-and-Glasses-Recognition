import os
import random

import dlib
import yaml
from tqdm import tqdm

from utils.classifier.feature_classifier import print_metrics, FeatureClassifier
from utils.object.TrainImage import TrainImage
from utils.utils import extract_features_for_training, Labeler, ImageType, Feature

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_features(config):
    return config['features']['with']


def get_image_with_feature(config, feature):
    path = config['dataset']['training'] + config['features']['with'][feature]
    img_type = ImageType.FEATURE.value
    return {'path': path, 'type': img_type}


def get_image_without_feature(config, feature):
    path = config['dataset']['training'] + config['features']['without'][feature]
    img_type = ImageType.NO_FEATURE.value
    return {'path': path, 'type': img_type}


def get_model(config, feature):
    return config['models'][feature]


def get_predictor_path(config):
    return config['libs']['predictor']


def generate_empty_dataset():
    return {'train': {0: [], 1: []}, 'val': {0: [], 1: []}}


def generation_datasets(images_with_feature, images_without_feature, train_val_ratio, kind_of_feature):
    src = [images_without_feature, images_with_feature]
    desc_train = ["Fill training set of No ", "Fill training set of "]
    desc_val = ["Fill validation set of No ", "Fill validation set of "]
    dataset = generate_empty_dataset()

    # Fill training and validation set
    for source in src:
        folder_path = source['path']
        files = os.listdir(folder_path)
        has_feature = source['type']
        number_of_files = int(len(files) * train_val_ratio)

        for file_name in tqdm(random.sample(files, number_of_files),
                              desc=desc_train[has_feature] + str(kind_of_feature)):
            dataset['train'][has_feature].append(TrainImage(file_name, folder_path))

        for file_name in tqdm(files, desc=desc_val[has_feature] + str(kind_of_feature)):
            if not any(train_image.is_equal(file_name) for train_image in dataset['train'][has_feature]):
                dataset['val'][has_feature].append(TrainImage(file_name, folder_path))

    return dataset


def generation_features(images_with_feature, images_without_feature, train_val_ratio, predictor, detector,
                        kind_of_feature):
    print("\nDataset...")
    dataset_images = generation_datasets(images_with_feature,
                                         images_without_feature,
                                         train_val_ratio,
                                         kind_of_feature)
    print("\nLabeler...")
    labeler = Labeler(dataset_images)
    print("\nFeature...")
    return extract_features_for_training(dataset_images,
                                         labeler,
                                         detector,
                                         predictor,
                                         kind_of_feature)


def training(X_train, y_train, X_val, y_val, fitted_model_path, metrics=True):
    model = FeatureClassifier()
    print("\nFit...")
    y_pred = model.fit(X_train, y_train, X_val)
    if metrics:
        print_metrics(y_val, y_pred)
    model.save(fitted_model_path)
    return y_pred


if __name__ == '__main__':

    with open('config.yml') as file:
        config = yaml.full_load(file)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(get_predictor_path(config))
    features = get_features(config)
    train_val_ratio = 70 / 100

    for feature in features:
        images_with_feature = get_image_with_feature(config, feature)
        images_without_feature = get_image_without_feature(config, feature)
        fitted_model = get_model(config, feature)
        kind_of_feature = Feature(feature)

        if not os.path.exists(fitted_model):
            os.makedirs(fitted_model)

        # Generation feature
        X_train, y_train, X_val, y_val = generation_features(images_with_feature,
                                                             images_without_feature,
                                                             train_val_ratio,
                                                             predictor,
                                                             detector,
                                                             kind_of_feature)
        # Fitting and prediction
        y_prediction = training(X_train,
                                y_train,
                                X_val,
                                y_val,
                                fitted_model)
