from cellsmap.util import load_movie

def test_load_movie():
    movie = load_movie('cdh5_dir', time_end = 2)
    assert movie.shape[0] == 3

    movie = load_movie('cdh5_dir', time_start=1, time_end = 2)
    assert movie.shape[0] == 2

    