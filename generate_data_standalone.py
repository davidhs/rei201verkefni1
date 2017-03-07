import numpy as np
import scipy.ndimage  # notum þetta fyrir myndir
import glob  # Notað til að finna skrár.

from prog import *  # Allt draslið er hér!

# HLUTI 1
albums = [load_image_as_arr(path) for path in glob.glob('albums/*.jpg')]

# HLUTI 2
A_images = np.load("mnist/A_images.npy")
A_labels = np.load("mnist/A_labels.npy")
B_images = np.load("mnist/B_images.npy")
B_labels = np.load("mnist/B_labels.npy")

# Búum hér sífellt til bundle og vistum út í skrá
while True:
    generate_data_part1(albums)
    generate_data_part2(A_images)
