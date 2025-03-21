import numpy as np
import pandas as pd

def save_train_test(train_test_dict, savedir) -> None:
    '''Save train test data to file in savedir, using numpy savez.'''
    np.savez(savedir+'train_test_data', **train_test_dict)

def load_train_test(file_path:str) -> dict:
    '''Load train test data from file_path.'''
    return dict(np.load(file_path, allow_pickle=True))

def save_model(model_dict:dict, savedir:str) -> None:
    '''Save model to file in savedir.'''
    np.savez(savedir+'drift_diffusion_model',**model_dict)

def load_model(file_path:str) -> dict:
    '''Load model from file_path.'''
    return dict(np.load(file_path, allow_pickle=True))
