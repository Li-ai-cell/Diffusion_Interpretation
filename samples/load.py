import random
import numpy as np
from PIL import Image

# Load the .npz file

def main():
    filename = 'samples_100x256x256x3.npz'
    data = np.load(filename)

    random_idx = random.sample(range(100), 5)

    for idx in random_idx:

        img = Image.fromarray(data['arr_0'][idx])
        # Save the image
        img.save(f'imgs/output_image_{idx}.png')


if __name__ == "__main__":
    main()