import os

SDCIT_WORKING_DIR = os.path.expanduser('~/Dropbox/research/2014 rcm/workspace/python/SDCIT/UAI_2017_SDCIT_experiments')
SDCIT_RESULT_DIR = SDCIT_WORKING_DIR + '/results'
SDCIT_FIGURE_DIR = SDCIT_WORKING_DIR + '/figures'
SDCIT_DATA_DIR = os.path.expanduser('~/kcipt_data')

assert os.path.exists(SDCIT_WORKING_DIR), 'working directory:{} does not exist'.format(SDCIT_WORKING_DIR)
if not os.path.exists(SDCIT_FIGURE_DIR):
    os.makedirs(SDCIT_FIGURE_DIR)
if not os.path.exists(SDCIT_RESULT_DIR):
    os.makedirs(SDCIT_RESULT_DIR)
if not os.path.exists(SDCIT_DATA_DIR):
    os.makedirs(SDCIT_DATA_DIR)
