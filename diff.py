import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from PIL import Image

def diff(truthFilename, filename) -> None:
    truth = np.array(Image.open(truthFilename))
    image = np.array(Image.open(filename))
    diff = 255 - np.abs(truth - image)
    plt.imsave(filename[:len(filename) - 4] + "_diff.png", diff)

truth = "./img/denoise_small_light_ground_truth.png"
images = ["./img/denoise_small_light_100_iters.png"]
for file in images:
    diff(truth, file)