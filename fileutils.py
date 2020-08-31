import os
import random
import zipfile
import csv
import numpy as np
from shutil import copyfile

'''
    Functions in file:
        -- csv_split - Split given csv into data and label.
        -- unzip - To unzip and extract files from .zip file to given location.
        -- make_directories - To create directories given listof path names.
        -- split_data - Create training/test split into respective folders given source and split ratio.
        
'''

def csv_split(filename,col_num):
    '''Given a CSV File path and the number of columns in the file, Return data and label split.
    (Here label is assumed to be at column 0.)'''
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file, delimiter=',')
        first_line = True
        temp_images = []
        temp_labels = []
        for row in csv_reader:
            if first_line:
                # print("Ignoring first line")
                first_line = False
            else:
                temp_labels.append(row[0])
                image_data = row[1:col_num]
                image_data_as_array = np.array_split(image_data, 28)
                temp_images.append(image_data_as_array)
        images = np.array(temp_images).astype('float')
        labels = np.array(temp_labels).astype('float')
    return images, labels

def unzip(zip_location, extract_location):
    ''' -- Given zip file location and extract location, Contents from zipfile will be extracted to given location.'''
    zip_ref = zipfile.ZipFile(zip_location, 'r')
    zip_ref.extractall(extract_location)
    zip_ref.close()


def make_directories(list_of_directories):
    '''-- Given a list of directory path names, Directories will be created in those paths.'''
    for directory in list_of_directories:
        try:
            os.mkdir(directory)
            print(directory, 'Directory Created')
        except:
            print(directory, 'Creation Failed')


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    '''
    -- Given the following parameters:
        SOURCE: SOURCE directory containing the files
        TRAINING: TRAINING directory that a portion of the files will be copied to
        TESTING: TESTING directory that a portion of the files will be copie to
        SPLIT_SIZE: SPLIT SIZE to determine the portion

    -- A split of training and testing images with given ratio will be added into their respective folders.
    '''

    all_files = []

    for file_name in os.listdir(SOURCE):
        file_path = SOURCE + file_name

        if os.path.getsize(file_path):
            all_files.append(file_name)
        else:
            print('{} is zero length, so ignoring'.format(file_name))

    n_files = len(all_files)
    split_point = int(n_files * SPLIT_SIZE)

    shuffled = random.sample(all_files, n_files)

    train_set = shuffled[:split_point]
    test_set = shuffled[split_point:]

    for file_name in train_set:
        copyfile(SOURCE + file_name, TRAINING + file_name)

    for file_name in test_set:
        copyfile(SOURCE + file_name, TESTING + file_name)
