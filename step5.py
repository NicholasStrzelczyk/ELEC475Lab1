import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import step4


def get_user_input():
    print("~~ Step 5: Image De-noising ~~")
    print("To select an image, please enter an integer between 0 and 59999: ")
    user_input = int(input())
    if user_input < 0 | user_input > 59999:
        print("Error: integer is out of bounds! - quitting program...")
        return -1
    return user_input


def get_reference_image(idx):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST('./data/mnist', train=False, download=True, transform=transform)
    return dataset.data[idx]


def get_noisy_image(img):
    noisy_img = img / 255
    noisy_img = noisy_img + torch.rand(img.size())
    return noisy_img


def get_output_image(input_img, model):
    with torch.no_grad():
        img = torch.flatten(input_img, start_dim=0)
        output_img = model.forward(img)
        return np.reshape(output_img, (28, 28))


def display_images(source_img, noisy_img, output_img):
    f = plt.figure()
    f.add_subplot(1, 3, 1)
    plt.imshow(source_img, cmap='gray')
    f.add_subplot(1, 3, 2)
    plt.imshow(noisy_img, cmap='gray')
    f.add_subplot(1, 3, 3)
    plt.imshow(output_img, cmap='gray')
    plt.show()


def run(model):
    # get valid user input integer
    user_idx = get_user_input()
    if user_idx == -1:
        quit()

    # get the reference & output images
    source_img = get_reference_image(user_idx)
    noisy_img = get_noisy_image(source_img)
    output_img = get_output_image(noisy_img, model)

    # display the image and label for the given index
    display_images(source_img, noisy_img, output_img)

    return 0
