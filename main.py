import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import model


def get_reference_image(idx):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST('./data/mnist', train=False, download=True, transform=transform)
    return dataset.data[idx]


def get_output_image(idx, model):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST("./data/mnist", train=False, download=True, transform=transform)
    tensor, _ = dataset[idx]
    loader = DataLoader(tensor, batch_size=2048, shuffle=False)
    with torch.no_grad():
        for img in loader:
            img = torch.flatten(img, start_dim=1)
            img = model.forward(img)
            output_img = np.reshape(img, (28, 28))
    return output_img


def get_user_input():
    print("Please enter an integer between 0 and 59999: ")
    user_input = int(input())
    if user_input < 0 | user_input > 59999:
        print("ERROR: integer is out of bounds!")
        return -1
    return user_input


def display_images(source_img, output_img):
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(source_img, cmap='gray')
    f.add_subplot(1, 2, 2)
    plt.imshow(output_img, cmap='gray')
    plt.show()


def main():
    # get user input integer
    user_idx = get_user_input()

    # make sure user input is valid
    if user_idx == -1:
        quit()

    # instantiate trained model & generate output
    my_model = model.AutoencoderMlp4Layer()
    my_model.load_state_dict(torch.load("MLP.8.pth"))
    my_model.eval()

    # get the reference & output images
    source_img = get_reference_image(user_idx)
    output_img = get_output_image(user_idx, my_model)

    # display the image and label for the given index
    display_images(source_img, output_img)

    # end main
    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
