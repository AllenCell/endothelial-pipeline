from cellsmap.util import load_dataset

def test_load_dataset():
    movie = load_dataset('cdh5_dir', time_end = 2)
    assert movie.shape[0] == 3

    movie = load_dataset('cdh5_dir', time_start=1, time_end = 2)
    assert movie.shape[0] == 2

    