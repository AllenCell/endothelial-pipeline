from cellsmap.util import load_dataset
# the cellsmap.util import above takes approximately 1m 16s to complete,
# that seems awfull long, is that normal?
from matplotlib import pyplot as plt
from skimage import filters


movie_name = 'cdh5_path'
img_bin = 1
test = load_dataset(movie_name, time_start=0, time_end=10, resolution=img_bin)

test_arr = test[0, test.shape[1]//2-(512/(img_bin + 1)):test.shape[1]//2+(512/(img_bin + 1)),  test.shape[2]//2-(512/(img_bin + 1)):test.shape[2]//2+(512/(img_bin + 1))]
## takes about 50s to compute the image subset
test_arr = test_arr.compute()

test_gauss = filters.gaussian(test_arr)

plt.imshow(test_arr, vmax=200)






