# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file and neuralnet_part2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class NeuralNet(torch.nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initialize the layers of your neural network
        Consruct network structure
        
        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        We recommend setting the lrate to 0.01 for part 1

        nn.Linear() uses a Kaiming He uniform initialization to initialize the weight matrices and 0 for the bias terms.
        nn.Sequential()
        torch.tensor()
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.fc1 = nn.Linear(in_size, 32)
        self.fc2 = nn.Linear(32, out_size)
        self.optimizer = optim.SGD(self.parameters(), lr=lrate, momentum=0.9)

    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        y = self.fc2(F.relu(self.fc1(x)))
        return y

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y)
        loss.backward()
        self.optimizer.step()
        return loss


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of iterations of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N
    """
    losses, yhats = [], []
    net = NeuralNet(0.01, torch.nn.CrossEntropyLoss(), len(train_set[0]), 2)
    stan_train = (train_set-torch.mean(train_set))/torch.std(train_set)
    stan_dev = (dev_set-torch.mean(dev_set))/torch.std(dev_set)

    # training
    for epoch in range(n_iter):  # loop over the dataset multiple times
        # get the inputs; data is a list of [inputs, labels]
        inputs = stan_train[(batch_size*epoch)%train_set.shape[0]:batch_size*(epoch+1)]
        labels = train_labels[(batch_size*epoch)%train_set.shape[0]:batch_size*(epoch+1)]

        loss = net.step(inputs, labels)

        # print statistics
        losses.append(loss.item())
    
    net.eval()
    # testing
    for img in stan_dev:
        y = torch.argmax(net.forward(img))
        yhats.append(y.item())
    
    yhats = np.array(yhats)
    
    assert(len(yhats) == dev_set.shape[0])
    return losses, yhats, net
