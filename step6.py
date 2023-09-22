import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import model


def get_user_inputs():
    print("~~ Step 6: Interpolation ~~")
    inputs = {}

    print("1. To select a start image, please enter an integer between 0 and 59999: ")
    user_input = int(input())
    if user_input < 0 | user_input > 59999:
        print("Error: integer is out of bounds! - quitting program...")
        return -1
    inputs[0] = user_input

    print("2. To select a end image, please enter an integer between 0 and 59999: ")
    user_input = int(input())
    if user_input < 0 | user_input > 59999:
        print("Error: integer is out of bounds! - quitting program...")
        return -1
    inputs[1] = user_input

    print("3. Please specify the number of interpolations (integer between 1 and 9): ")
    user_input = int(input())
    if user_input < 1 | user_input > 9:
        print("Error: integer is out of bounds! - quitting program...")
        return -1
    inputs[2] = user_input

    return inputs


def get_reference_image(idx):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST('./data/mnist', train=False, download=True, transform=transform)
    return dataset.data[idx]


def get_lerp_weight(n_lerps):
    # custom weight function based on number of lerps
    return float((n_lerps / (1.5 * n_lerps * n_lerps)) + 0.15)


def get_interpolations_array(start_img, end_img, n_lerps,  model):
    # initialize array and weight parameter
    interpolations = {}
    lerp_weight = get_lerp_weight(n_lerps)

    # make sure gradient calculation is disabled
    with torch.no_grad():
        # flatten and encode start and end images
        start_img = torch.flatten(start_img, start_dim=0) / 255
        start_img = model.encode(start_img)

        end_img = torch.flatten(end_img, start_dim=0) / 255
        end_img = model.encode(end_img)

        # populate start and end indexes of the array
        interpolations[0] = start_img
        interpolations[n_lerps+1] = end_img

        # process n steps of interpolation using bottleneck tensors
        for i in range(1, n_lerps+1):
            temp_tensor = torch.Tensor.new(start_img)
            torch.lerp(interpolations[i-1], end_img, lerp_weight, out=temp_tensor)
            interpolations[i] = temp_tensor

        # decode every bottleneck tensor in the array
        for i in range(len(interpolations)):
            temp = model.decode(interpolations[i])
            interpolations[i] = np.reshape(temp, (28, 28))

    return interpolations


def display_images(interpolations):
    f = plt.figure()
    for i in range(len(interpolations)):
        f.add_subplot(1, len(interpolations), i+1)
        plt.imshow(interpolations[i], cmap='gray')
    plt.show()


def run(model):
    # get valid user input integer
    user_inputs = get_user_inputs()
    if user_inputs[0] == -1 | user_inputs[1] == -1 | user_inputs[2] == -1:
        quit()

    # extract interpolation parameters
    start_img = get_reference_image(user_inputs[0])
    end_img = get_reference_image(user_inputs[1])
    n_lerps = user_inputs[2]

    # generate the tensor array with interpolations
    img_array = get_interpolations_array(start_img, end_img, n_lerps, model)

    # display the resulting images
    display_images(img_array)

    return 0


if __name__ == "__main__":
    my_model = model.AutoencoderMlp4Layer()
    my_model.load_state_dict(torch.load("MLP.8.pth"))
    my_model.eval()
    run(my_model)
