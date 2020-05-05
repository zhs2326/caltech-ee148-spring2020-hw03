from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(8, 16, 5, 1)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout1 = nn.Dropout(0.5)
        self.last = []

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        #X = F.sigmoid(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        #X = F.sigmoid(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        #if you want to visualize the output of the last fc
        visual = False
        if visual:
            x_np = x.numpy()
            self.last.extend(x_np)


        output = F.log_softmax(x, dim=1)


        return output


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    correct = 0
    loss_epoch = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step

        correct += (torch.argmax(output, dim=1) == target).float().sum()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))

        loss_epoch.append(loss.item())

    accuracy = (100 * correct / len(train_loader.sampler)).item()
    print('Accuracy on training set: {:.0f}'.format(accuracy)+' '+str(correct.item())+'/'+str(len(train_loader.sampler)))

    with open('trian_acc.txt', 'a') as file:
        file.write("{:.0f}".format(accuracy) + ',')

    with open('trian_loss.txt', 'a') as file:
        file.write(str(sum(loss_epoch)/len(loss_epoch))+',')


def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    wrong_examples = []
    y_true = []
    y_pred = []
    global y_out
    y_out = []

    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            y_pred.extend(pred)
            y_true.extend(target)
            y_out.extend(target)

            for i, res in enumerate(pred.eq(target.view_as(pred))):
                if not res:
                    wrong_examples.append(i)

            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    if test_num:
        test_loss /= test_num

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, test_num,
            100. * correct / test_num))

    #if you want to show the wrong example
    #print('wrong examples', wrong_examples)

    #if you want to show the confusion matrix
    #print(confusion_matrix(y_true, y_pred))

    if test_num:
        with open('test_acc.txt', 'a') as file:
            file.write("{:.0f}".format(100. * correct / test_num) + ',')

    with open('test_loss.txt', 'a') as file:
        file.write(str(test_loss) + ',')


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        #if you want to show the kernels
        kernels = model.state_dict()['conv1.weight']
        print(kernels)

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        '''
        subset_indices_train = []
        for i in range(len(test_dataset.classes)):
            idx = (test_dataset.targets == i)
            idx = [id for id, x in enumerate(idx) if x]
            subset_indices_train.extend(np.random.choice(idx, int(0.85 * len(idx)), replace=False))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, sampler=SubsetRandomSampler(subset_indices_train))
        '''


        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)




        test(model, device, test_loader)

        #visualize tsne
        '''
        last = np.array(model.last)
        last_embedded = TSNE(n_components=2).fit_transform(last)
        vis_x = last_embedded[:, 0]
        vis_y = last_embedded[:, 1]
        plt.scatter(vis_x, vis_y, c=y_out, cmap=plt.cm.get_cmap("jet", 10))
        plt.colorbar(ticks=range(10))
        plt.clim(-0.5, 9.5)
        plt.show()
        '''

        #find near images
        '''
        rand_images_indices = np.random.choice(range(10000), 4)
        near_image_dict = {}
        '''

        #change shuffle to False
        '''
        for image_index in rand_images_indices:
            x_image = last[image_index]
            dist_l = []
            for other_index in range(10000):
                if other_index != image_index:
                    other_image = last[other_index]
                    dist_l.append((np.linalg.norm(x_image-other_image), other_index))
            dist_l.sort()
            #print(dist_l)
            near_image_dict[image_index] = [other[1] for other in dist_l[:8]]
        print('near_image_dict', near_image_dict)
        '''





        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    #transforms.RandomResizedCrop((28, 28), (0.9, 1.0)),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.RandomRotation(30),
                    transforms.ToTensor(),           # Add data augmentation here
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    val_dataset = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose([  # Data preprocessing
                                       transforms.ToTensor(),  # Add data augmentation here
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
    subset_indices_train = []
    for i in range(len(train_dataset.classes)):
        idx = (train_dataset.targets == i)
        idx = [id for id, x in enumerate(idx) if x]
        subset_indices_train.extend(np.random.choice(idx, int(0.85*len(idx)), replace = False))

    subset_indices_valid = np.setdiff1d(range(len(train_dataset)), subset_indices_train)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)
    print (sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, val_loader)
        scheduler.step()    # learning rate scheduler

        # You may optionally save your model at each epoch here

    if args.save_model:
        torch.save(model.state_dict(), "mnist_model.pt")


if __name__ == '__main__':
    main()
