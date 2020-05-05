from PIL import Image
import torch
from torchvision import datasets

test_dataset = datasets.MNIST('../data', train=False)
wrong_examples = [62, 247, 320, 321, 445, 449, 619, 674, 813]

background = Image.new('RGBA',(100, 100), (255, 255, 255, 255))

for i, example in enumerate(wrong_examples):
    img = test_dataset[example][0]
    offset = (5+i//3*30, 5+(i%3)*30)
    background.paste(img, offset)

background.show()