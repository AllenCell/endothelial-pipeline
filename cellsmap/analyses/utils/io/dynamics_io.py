import numpy as np
import pickle

def save_train_test(train_test_dict:dict, savedir:str) -> None:
    '''Save train test data to file in savedir, using numpy savez.'''
    np.savez(savedir+'train_test_data', **train_test_dict)

def load_train_test(file_path:str) -> dict:
    '''Load train test data from file_path.'''
    return dict(np.load(file_path, allow_pickle=True))

def save_model(model_dict:dict, savedir:str) -> None:
    '''Save model to file in savedir.'''
    with open(savedir+'drift_diffusion_model.pkl', 'wb') as f:
        pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(file_path:str) -> dict:
    '''Load model from file_path.'''
    # check if pysindy is imported, if not, import it
    if 'ps' or 'pysindy' not in globals():
        import pysindy as ps
    with open(file_path, 'rb') as f:
        return pickle.load(f)
