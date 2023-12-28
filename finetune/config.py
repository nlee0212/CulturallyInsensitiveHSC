# Standard
import random

# PIP
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

# Custom


class Config:
    # User Setting
    SEED = 94

    model = 'xlm-roberta-base' # model name for finetuning
    if_arabic = False 
    max_length = 128

    checkpoint_filename = CHECKPOINT_FILENAME
    best_filename = BEST_CHECKPOINT_FILENAME
    additional_tokens = []
    remove_special_tokens = True

    train_data = PATH_TO_TRAIN_DATA_CSV
    val_data = PATH_TO_VALID_DATA_CSV

    sent_col = COLUMN_NAME_WITH_TEXTS
    label_col = COLUMN_NAME_WITH_LABELS
    num_labels = 2

    test_data = PATH_TO_TEST_DATA_CSV
    test_res = PATH_TO_TEST_DATA_RESULT_CSV # inference results will be saved

    test_col = COLUMN_NAME_WITH_TEXTS_IN_TEST_FILE
    test_res_col = COLUMN_NAME_TO_SAVE_LABELS_IN_TEST_RES_FILE
    
    batch_size = 32
    num_workers = 4
    distributed=True

    train = True
    evaluate = True
    resume = True
    resume_model =best_filename

    # test = True
    test = False

    start_epoch = 0
    epochs = 5



    def __init__(self, SEED=None):
        if SEED:
            self.SEED = SEED
        self.set_random_seed()

    def set_random_seed(self):
        print(f'=> SEED : {self.SEED}')

        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(self.SEED)  # if use multi-GPU

        pl.seed_everything(self.SEED)
# 
