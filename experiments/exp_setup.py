import multiprocessing
import os

SDCIT_WORKING_DIR = os.path.expanduser('~/Downloads/SDCIT/experiments')
SDCIT_RESULT_DIR = SDCIT_WORKING_DIR + '/results'
SDCIT_FIGURE_DIR = SDCIT_WORKING_DIR + '/figures'
SDCIT_DATA_DIR = os.path.expanduser('~/kcipt_data')

PARALLEL_JOBS = multiprocessing.cpu_count() // 2

assert os.path.exists(SDCIT_WORKING_DIR), 'working directory:{} does not exist. Please set appropriately'.format(SDCIT_WORKING_DIR)
if not os.path.exists(SDCIT_FIGURE_DIR):
    print('creating directory: {}'.format(SDCIT_FIGURE_DIR))
    os.makedirs(SDCIT_FIGURE_DIR)
if not os.path.exists(SDCIT_RESULT_DIR):
    print('creating directory: {}'.format(SDCIT_RESULT_DIR))
    os.makedirs(SDCIT_RESULT_DIR)
if not os.path.exists(SDCIT_DATA_DIR):
    print('creating directory: {}'.format(SDCIT_DATA_DIR))
    os.makedirs(SDCIT_DATA_DIR)
    print('please checkout https://github.com/garydoranjr/kcipt and run our MATLAB code to generate data used in the paper.')
