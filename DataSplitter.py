import os
import shutil
from os.path import join

from sklearn.model_selection import train_test_split

from DataTypes import ChartType


def split_data():
    chart_types = [
        ChartType('./original_data/chimeric', './train_data/chimeric', './test_data/chimeric'),
        ChartType('./original_data/regular', './train_data/non_chimeric', './test_data/non_chimeric'),
        ChartType('./original_data/repeat', './train_data/non_chimeric', './test_data/non_chimeric')
    ]

    # Clear existing training & test data
    for ct in chart_types:
        shutil.rmtree(ct.train_path)
        shutil.rmtree(ct.test_path)

        os.mkdir(ct.train_path)
        os.mkdir(ct.test_path)

    for ct in chart_types:
        # List all original files in folder at path
        filenames = os.listdir(ct.original_path)
        # Split files for training & test
        xs_train, xs_test = train_test_split(filenames)

        # Copy training files to matching train_data folder
        for x in xs_train:
            full_file_path = join(ct.original_path, x)
            shutil.copy(src=full_file_path, dst=ct.train_path)

        # Copy test files to matching test_data folder
        for x in xs_test:
            full_file_path = join(ct.original_path, x)
            shutil.copy(src=full_file_path, dst=ct.test_path)
