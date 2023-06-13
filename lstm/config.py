import configparser
import os
import json

config = configparser.ConfigParser()
config_file_path = os.path.join(os.path.dirname(__file__),'..', 'config.ini')
config.read(config_file_path)


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints')

BATCH_SIZE = 1
EPOCHS = 100

RC_DATA_PATH = os.path.join(DATA_DIR,'rcs.npy')
GT_DATA_PATH = os.path.join(DATA_DIR,'gts.npy')

RC_TEST_DATA_PATH = os.path.join(DATA_DIR,'rc_tests_input.npy')
GT_TEST_DATA_PATH = os.path.join(DATA_DIR,'gt_tests_input.npy')

VOI_SIZE = [201, 101, 101]

# Overload values from config
if os.path.isfile(config_file_path):
    DATA_DIR = config.get('DATA', 'DATA_DIR', fallback=DATA_DIR)
    RC_DATA_PATH = config.get('DATA', 'RC_TRAIN_DATA_PATH', fallback=RC_DATA_PATH)
    GT_DATA_PATH = config.get('DATA', 'GT_TRAIN_DATA_PATH', fallback=GT_DATA_PATH)

    CHECKPOINT_DIR = config.get('TRAIN', 'CHECKPOINT_DIR', fallback=CHECKPOINT_DIR)
    BATCH_SIZE = config.getint('TRAIN','BATCH_SIZE', fallback=BATCH_SIZE)
    EPOCHS = config.getint('TRAIN','EPOCHS', fallback=EPOCHS)

    RC_TEST_DATA_PATH = config.get('DATA', 'RC_TEST_DATA_PATH', fallback=RC_TEST_DATA_PATH)
    GT_TEST_DATA_PATH = config.get('DATA', 'GT_TEST_DATA_PATH', fallback=GT_TEST_DATA_PATH)
    
    VOI_SIZE = json.loads(config.get('DEFAULT', 'VOI_SIZE', fallback=VOI_SIZE))
