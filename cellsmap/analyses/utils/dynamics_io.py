import numpy as np
import pandas as pd

def save_train_test(train_test_dict, savedir) -> None:
    '''Save train test data to file in savedir, using numpy savez.'''
    np.savez(savedir+'train_test_data', **train_test_dict)

def load_train_test(file_path:str) -> dict:
    '''Load train test data from file_path.'''
    return np.load(file_path, allow_pickle=True)

