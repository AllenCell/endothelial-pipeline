from cellsmap.util import load_dataset
from cellpose import models


def test_load_dataset():
    # check end point specification
    movie = load_dataset('cdh5_path', time_end = 2)
    assert movie.shape[0] == 3
    # check start point specification
    movie = load_dataset('cdh5_path', time_start=1, time_end = 2)
    assert movie.shape[0] == 2
    # check resolution specification
    movie = load_dataset('cdh5_path', time_start=1, time_end = 2, resolution=1)
    assert movie.shape[1:] == (856, 4796)

    
if __name__ == '__main__':

    dat = test_load_dataset()
    
