import argparse
import os
import shutil
import subprocess
import sys

from cv2 import cv2

from utils.utils import Feature


def save_to(path, images_indexes, dataset, dataset_name, feature_tag):
    for idx in tqdm(images_indexes, desc=f'Save {feature_tag}'):
        cv2.imwrite(f'{path}/{dataset_name}_{idx}.jpg', dataset[idx])


def save_dataset_images(dataset, dataset_name, path, feature_tag, no_feature_tag, all_indexes, feature_indexes,
                        noFeature_indexes):
    save_to(f"{path}{feature_tag}", feature_indexes, dataset, dataset_name, feature_tag)
    save_to(f"{path}{no_feature_tag}", noFeature_indexes, dataset, dataset_name, no_feature_tag)


def get_features_dict(hdf):
    results = {Feature.BEARD.value: [], Feature.MUSTACHE.value: [], Feature.GLASSES.value: []}
    nothing = []
    for idx, row in enumerate(hdf):
        if row[0] == 1:
            results[Feature.BEARD.value].append(idx)
        elif row[0] == 2:
            results[Feature.MUSTACHE.value].append(idx)

        if row[1] == 1:
            results[Feature.GLASSES.value].append(idx)

        if row[0] == 0 and row[1] == 0:
            nothing.append(idx)

    return results, nothing


def generate_folders(path, folders_names):
    for key in folders_names:
        new_path = path + folders_names[key]
        if os.path.exists(new_path):
            shutil.rmtree(new_path)

        os.makedirs(new_path)


def prepare_training_set(config):
    path = config['dataset']['training']
    features = config['features']['with']
    no_features = config['features']['without']

    generate_folders(path, features)
    generate_folders(path, no_features)

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".h5"):
                with h5py.File(f"{path}{file}", 'r') as hdf:
                    dictionary, nothing = get_features_dict(zip(hdf['y_bm'], hdf['y_g']))
                    dataset_name = file.rpartition('.')[0]
                    for key in dictionary.keys():
                        save_dataset_images(hdf['X'],
                                            dataset_name,
                                            path,
                                            features[key],
                                            no_features[key],
                                            range(len(hdf['y_bm'])),
                                            dictionary[key],
                                            nothing)

                os.remove(f"{path}{file}")


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--user"])


def import_elements_from(drive_ids, path, ext):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)
    for id in drive_ids:
        url = f'https://drive.google.com/uc?id={id}'
        output = f'{path}/{id}{ext}'
        gdown.download(url, output, quiet=False)

        if ext == '.zip':
            shutil.unpack_archive(output, path)
            os.remove(output)


def init_parameter():
    parser = argparse.ArgumentParser(description='This script help configure the project')
    parser.add_argument("--models",
                        default='False',
                        choices=('True', 'False'),
                        help="Import the models")
    parser.add_argument("--training",
                        default='False',
                        choices=('True', 'False'),
                        help="Import the training set")
    parser.add_argument("--libs",
                        default='True',
                        choices=('True', 'False'),
                        help="Import the library")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    modules = ['gdown', 'pyyaml', 'h5py', 'tqdm']

    for module in modules:
        if module not in sys.modules:
            install(module)

    try:
        import gdown
        import yaml
        import h5py
        from tqdm import tqdm
    except Exception as e:
        print('Import failed:', e)
        sys.exit()

    args = init_parameter()

    with open('config.yml') as file:
        config = yaml.full_load(file)

    for configuration in config['configurations']:
        if config['configurations'][configuration]['import'] or (vars(args)[configuration] == 'True'):
            drive_ids = config['configurations'][configuration]['id']
            import_path = config['configurations'][configuration]['path']
            ext = config['configurations'][configuration]['ext']
            import_elements_from(drive_ids, import_path, ext)

    if config['configurations']['training']['import'] or (vars(args)['training'] == 'True'):
        prepare_training_set(config)
