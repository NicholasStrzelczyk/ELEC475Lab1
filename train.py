from datetime import datetime

import torch
import matplotlib.pyplot as plt

from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchsummary import summary
from argparse import ArgumentParser

import model


def train(n_epochs, model, optimizer, loss_fn, train_loader, scheduler, device='cpu', save_file=None, plot_file=None):
    print("training ... ")
    model.train()

    if save_file is not None:
        torch.save(model.state_dict(), save_file)

    losses_train = []

    for epoch in range(1, n_epochs + 1):
        print("epoch ", epoch)
        curr_loss_train = 0.0

        for images, labels in train_loader:
            images = torch.flatten(images, start_dim=1)
            images = images.to(device=device)
            outputs = model(images)
            img_loss = loss_fn(outputs, images)
            optimizer.zero_grad()
            img_loss.backward()
            optimizer.step()
            curr_loss_train += img_loss.item()

        scheduler.step(curr_loss_train)

        losses_train += [curr_loss_train / len(train_loader)]

        print("{} Epoch {}, Training loss {}".format(datetime.now(), epoch, curr_loss_train / len(train_loader)))

    if save_file is not None:
        torch.save(model.state_dict(), save_file)

    plt.figure(figsize=(12, 7))
    plt.clf()
    plt.plot(losses_train, label='Train')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if plot_file is not None:
        plt.savefig(plot_file)
    plt.show()


def main():
    # Run the following line in terminal to train
    # python train.py -z 8 -e 50 -b 2048 -s MLP.8.pth -p loss.MLP.8.png

    # set up arg parser
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-z', '--bottleneck')
    arg_parser.add_argument('-e', '--n_epochs')
    arg_parser.add_argument('-b', '--batch_size')
    arg_parser.add_argument('-s', '--saved_model_path')
    arg_parser.add_argument('-p', '--saved_plot_path')
    args = arg_parser.parse_args()

    # parse args from command line
    bottleneck = int(args.bottleneck)
    n_epochs = int(args.n_epochs)
    batch_size = int(args.batch_size)
    save_model_path = args.saved_model_path
    plot_file_path = args.saved_plot_path

    # initializes training data
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = MNIST("./data/mnist", train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # initialize model and parameters
    my_model = model.AutoencoderMlp4Layer(n_bottleneck=bottleneck)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=my_model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.1, last_epoch=-1)

    # run torch summary report
    summary(my_model, (1, 392, 784))

    ######################################################
    # call the train function using the parameters above #
    ######################################################
    train(
        n_epochs=n_epochs,
        model=my_model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        scheduler=scheduler,
        save_file=save_model_path,
        plot_file=plot_file_path
    )


if __name__ == '__main__':
    main()
