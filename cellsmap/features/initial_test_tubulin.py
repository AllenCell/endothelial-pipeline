from cellsmap.util import load_dataset
# the cellsmap.util import above takes approximately 1m 16s to complete,
# that seems awfull long, is that normal?
from matplotlib import pyplot as plt


movie_name = 'cdh5_path'
test = load_dataset(movie_name, time_start=0, time_end=10, resolution=1)

test_arr = test[0, test.shape[1]//2-512:test.shape[1]//2+512,  test.shape[2]//2-512:test.shape[1]//2+512]
## takes about 50s to compute the image subset
test_arr = test_arr.compute()

plt.imshow(test_arr, vmax=200)



