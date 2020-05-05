from PIL import Image
import torch
from torchvision import datasets

test_dataset = datasets.MNIST('../data', train=False)

#{263: [2563, 2844, 6825, 3572, 8727, 3993, 1799, 1133], 9769: [5856, 6671, 8920, 6249, 5289, 8796, 634, 1322], 1648: [9054, 5311, 5299, 2438, 4333, 9028, 7786, 4316], 8624: [7551, 9965, 8867, 2255, 5054, 8435, 3977, 5740]}
near_image_dict = {263: [2563, 2844, 6825, 3572, 8727, 3993, 1799, 1133], 9769: [5856, 6671, 8920, 6249, 5289, 8796, 634, 1322], 1648: [9054, 5311, 5299, 2438, 4333, 9028, 7786, 4316], 8624: [7551, 9965, 8867, 2255, 5054, 8435, 3977, 5740]}

background = Image.new('RGBA', (270, 125), (255, 255, 255, 255))

n = 0
for k, v in near_image_dict.items():
    images = [test_dataset[k]]
    for t in v:
        images.append(test_dataset[t])

    for i, img in enumerate(images):
        img = img[0]
        offset = (5 + i % 9 * 29, 5 + n * 29)
        background.paste(img, offset)

    n += 1
background.show()