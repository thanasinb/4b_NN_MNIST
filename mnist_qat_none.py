from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import namedtuple
import copy
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_channels = 1
        self.conv1 = nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.flatten_shape = 4 * 4 * 50
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.flatten_shape)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x




class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8, min_val=None, max_val=None):
        x = quantize_tensor(x, num_bits=num_bits, min_val=min_val, max_val=max_val)
        x = dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None


# x = torch.tensor([1, 2, 3, 4]).float()
# print(FakeQuantOp.apply(x))


# ## Quantization Aware Training Forward Pass
def quantAwareTrainingForward(model, x, stats, vis=False, axs=None, sym=False, num_bits=8, act_quant=False, mnist=True):
    conv1weight = model.conv1.weight.data
    # model.conv1.weight.data = FakeQuantOp.apply(model.conv1.weight.data, num_bits)
    x = F.relu(model.conv1(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')

    # if act_quant:
    #     x = FakeQuantOp.apply(x, num_bits, stats['conv1']['ema_min'], stats['conv1']['ema_max'])

    x = F.max_pool2d(x, 2, 2)

    conv2weight = model.conv2.weight.data
    # model.conv2.weight.data = FakeQuantOp.apply(model.conv2.weight.data, num_bits)
    x = F.relu(model.conv2(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')

    # if act_quant:
    #     x = FakeQuantOp.apply(x, num_bits, stats['conv2']['ema_min'], stats['conv2']['ema_max'])

    x = F.max_pool2d(x, 2, 2)

    if mnist:
        x = x.view(-1, 4 * 4 * 50)
    else:
        x = x.view(-1, 1250)

    fc1weight = model.fc1.weight.data
    # model.fc1.weight.data = FakeQuantOp.apply(model.fc1.weight.data, num_bits)
    x = F.relu(model.fc1(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc1')

    # if act_quant:
    #     x = FakeQuantOp.apply(x, num_bits, stats['fc1']['ema_min'], stats['fc1']['ema_max'])

    x = model.fc2(x)

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc2')

    return F.log_softmax(x, dim=1), conv1weight, conv2weight, fc1weight, stats


# # Train using Quantization Aware Training
def trainQuantAware(args, model, device, train_loader, optimizer, epoch, stats, act_quant=False, num_bits=8,
                    mnist=True):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, conv1weight, conv2weight, fc1weight, stats = quantAwareTrainingForward(model, data, stats,
                                                                                       num_bits=num_bits,
                                                                                       act_quant=act_quant, mnist=mnist)

        model.conv1.weight.data = conv1weight
        model.conv2.weight.data = conv2weight
        model.fc1.weight.data = fc1weight

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        # if batch_idx % args["log_interval"] == 0:
        if (mnist and batch_idx == 999) or (not mnist and batch_idx == 499):
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.0f}%), batch_idx: {}, log_interval: {}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.item(),
                    correct, len(train_loader.dataset),
                           100. * correct / len(train_loader.dataset),
                    (batch_idx + 1), args["log_interval"]))
    return stats


def testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=8):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, conv1weight, conv2weight, fc1weight, _ = quantAwareTrainingForward(model, data, stats,
                                                                                       num_bits=num_bits,
                                                                                       act_quant=act_quant)

            model.conv1.weight.data = conv1weight
            model.conv2.weight.data = conv2weight
            model.fc1.weight.data = fc1weight

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def mainQuantAware(mnist=True):
    epochs = 10
    num_bits = 4
    if mnist:
        batch_size = 60
        test_batch_size = 60
    else:
        batch_size = 100
        test_batch_size = 100
    lr = 0.01
    momentum = 0.5
    seed = 1
    log_interval = 500
    save_model = True
    no_cuda = False

    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if mnist:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = datasets.CIFAR10(root='./dataCifar', train=True,
                                    download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True, num_workers=2)

        testset = datasets.CIFAR10(root='./dataCifar', train=False,
                                   download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                  shuffle=False, num_workers=2)

    model = Net(mnist=mnist).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    args = {}
    args["log_interval"] = log_interval
    stats = {}
    for epoch in range(1, epochs + 1):
        if epoch > 1:
            act_quant = True
        else:
            act_quant = False

        stats = trainQuantAware(args, model, device, train_loader, optimizer, epoch, stats, act_quant,
                                num_bits=num_bits, mnist=mnist)
        # testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=num_bits)

    if (save_model):
        if mnist:
            torch.save(model.state_dict(), "mnist_cnn.pt")
        else:
            torch.save(model.state_dict(), "cifar10_cnn.pt")

    return model, stats, device


model, old_stats, old_device = mainQuantAware(mnist=True)

# # ## Load Dataset
# kwargs = {'num_workers': 1, 'pin_memory': True}
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])),
#     batch_size=64, shuffle=True, **kwargs)

# # ## Test Quant Aware
# print(old_stats)

# log_interval = 500
# args = {}
# args["log_interval"] = log_interval
# q_model = copy.deepcopy(model)
# testQuantAware(args, q_model, old_device, test_loader, stats=old_stats, act_quant=True, num_bits=4)
