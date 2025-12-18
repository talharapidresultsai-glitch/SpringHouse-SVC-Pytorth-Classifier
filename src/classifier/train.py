#!/usr/bin/env python

# A sample training component that trains a simple scikit-learn decision tree model.
# This implementation works in File mode and makes no assumptions about the input file names.
# Input is specified as CSV with a data point in each row and the labels in the first column.

from __future__ import print_function

import json
import os
import sys
import traceback
from train_model import train_main
#from utils.general import LOGGER


# These are the paths to where SageMaker mounts things in your container.

prefix = '/opt/ml/'
input_root = prefix + 'input'
input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

# This algorithm has a single channel of input data called 'training'. Since we run in
# File mode, the input files are copied to the directory specified here.
channel_name='training'
training_path = os.path.join(input_path, channel_name)

# The function to execute the training.
def train():
    print('Starting the training.')
    try:
        # Read in any hyperparameters that the user passed with the training job
        with open(param_path, 'r') as tc:
            trainingParams = json.load(tc)
        train_main(**trainingParams)
        
        print('Training complete.')
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)

def log_path_info():
    input_contents = os.listdir(input_root)
    input_contents_str = ' '.join(input_contents)
    LOGGER.info('Training Input folders: ' + input_contents_str)
    if 'data' in input_contents:
        data_contents = os.listdir(os.path.join(input_root, 'data'))
        data_contents_str = ' '.join(data_contents)
        LOGGER.info('Training Data folders: ' + data_contents_str)

if __name__ == '__main__':
    #log_path_info()
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
