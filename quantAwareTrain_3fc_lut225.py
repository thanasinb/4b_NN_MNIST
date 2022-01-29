from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
                    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
lut_ideal_15 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0],   # 0
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,  1,  1,  1,  1,  1],   # 1
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,  1,  2,  2,  2,  2],   # 2
                      [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2,  2,  2,  3,  3,  3],   # 3
                      [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3,  3,  3,  3,  4,  4],   # 4
                      [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,  4,  4,  4,  5,  5],   # 5
                      [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4,  4,  5,  5,  6,  6],   # 6
                      [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5,  5,  6,  6,  7,  7],   # 7
                      [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,  6,  6,  7,  7,  8],   # 8
                      [0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6,  7,  7,  8,  8,  9],   # 9
                      [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7,  7,  8,  9,  9,  10],  # 10
                      [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7,  8,  9,  10, 10, 11],  # 11
                      [0, 1, 2, 2, 3, 4, 5, 6, 6, 7, 8,  9,  10, 10, 11, 12],  # 12
                      [0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9,  10, 10, 11, 12, 13],  # 13
                      [0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9,  10, 11, 12, 13, 14],  # 14
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]) # 15

lut_actual_15 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0],
                       [0, 1, 1, 1, 1, 1, 2, 2, 2, 2,  2,  3,  3,  3,  3,  3],
                       [0, 1, 2, 2, 2, 3, 3, 3, 4, 4,  4,  5,  5,  5,  5,  6],
                       [0, 2, 2, 3, 3, 3, 4, 4, 5, 5,  5,  6,  6,  7,  7,  7],
                       [0, 2, 3, 3, 4, 4, 5, 5, 6, 6,  6,  7,  7,  8,  8,  8],
                       [0, 2, 3, 3, 4, 5, 5, 6, 6, 7,  7,  8,  8,  9,  9,  9],
                       [0, 2, 3, 4, 4, 5, 6, 6, 7, 7,  8,  8,  9,  9,  10, 10],
                       [0, 3, 3, 4, 5, 5, 6, 7, 7, 8,  8,  9,  9,  10, 10, 11],
                       [0, 3, 3, 4, 5, 6, 6, 7, 8, 8,  9,  9,  10, 10, 11, 12],
                       [0, 3, 4, 4, 5, 6, 7, 7, 8, 9,  9,  10, 10, 11, 12, 12],
                       [0, 3, 4, 5, 5, 6, 7, 8, 8, 9,  10, 10, 11, 12, 12, 13],
                       [0, 3, 4, 5, 5, 6, 7, 8, 8, 9,  10, 11, 11, 12, 12, 13],
                       [0, 3, 4, 5, 6, 6, 7, 8, 9, 9,  10, 11, 12, 12, 13, 13],
                       [0, 3, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14],
                       [0, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 12, 13, 14, 14],
                       [0, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 12, 13, 14, 15]])

                         # [0,  1,  2,  3,  4,  5,  6,  7,   8,     9,    10,   11,   12,   13,   14,   15]])
lut_ideal_225 = np.array([[0,	0,	0,	0,	0,	0,	0,	0,	 0,	    0,	  0,	0,	  0,	0,	  0,	0],
                          [0,	1,	2,	3,	4,	5,	6,	7,	 8,	    9,	  10,	11,	  12,	13,	  14,	15],
                          [0,	2,	4,	6,	8,	10,	12,	14,	 16,	18,	  20,	22,	  24,	26,	  28,	30],
                          [0,	3,	6,	9,	12,	15,	18,	21,	 24,	27,	  30,	33,	  36,	39,	  42,	45],
                          [0,	4,	8,	12,	16,	20,	24,	28,	 32,	36,	  40,	44,	  48,	52,	  56,	60],
                          [0,	5,	10,	15,	20,	25,	30,	35,	 40,	45,	  50,	55,	  60,	65,	  70,	75],
                          [0,	6,	12,	18,	24,	30,	36,	42,	 48,	54,	  60,	66,	  72,	78,	  84,	90],
                          [0,	7,	14,	21,	28,	35,	42,	49,	 56,	63,	  70,	77,	  84,	91,	  98,	105],
                          [0,	8,	16,	24,	32,	40,	48,	56,	 64,	72,	  80,	88,	  96,	104,  112,	120],
                          [0,	9,	18,	27,	36,	45,	54,	63,	 72,	81,	  90,	99,	  108,	117,  126,	135],
                          [0,	10,	20,	30,	40,	50,	60,	70,	 80,	90,	  100,	110,  120,	130,  140,	150],
                          [0,	11,	22,	33,	44,	55,	66,	77,	 88,	99,	  110,	121,  132,	143,  154,	165],
                          [0,	12,	24,	36,	48,	60,	72,	84,	 96,	108,  120,	132,  144,	156,  168,	180],
                          [0,	13,	26,	39,	52,	65,	78,	91,	 104,	117,  130,	143,  156,	169,  182,	195],
                          [0,	14,	28,	42,	56,	70,	84,	98,	 112,   126,  140,	154,  168,	182,  196,	210],
                          [0,	15,	30,	45,	60,	75,	90,	105, 120,	135,  150,	165,  180,	195,  210,	225]])

                         # [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,    12,    13,    14,    15]])
lut_actual_225 = np.array([[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	 0,	    0,	   0,	  0,	 0],
                           [0,	1,	1,	1,	1,	1,	2,	2,	2,	2,	2,	 3,	    3,	   3,	  3,	 3],
                           [0,	1,	4,	4,	4,	6,	6,	6,	8,	8,	8,	 10,	10,	   10,	  10,	 12],
                           [0,	2,	4,	9,	9,	9,	12,	12,	15,	15,	15,	 18,	18,	   21,	  21,	 21],
                           [0,	2,	6,	9,	16,	16,	20,	20,	24,	24,	24,	 28,	28,	   32,	  32,	 32],
                           [0,	2,	6,	9,	16,	25,	25,	30,	30,	35,	35,	 40,	40,	   45,	  45,	 45],
                           [0,	2,	6,	12,	16,	25,	36,	36,	42,	42,	48,	 48,	54,	   54,	  60,	 60],
                           [0,	3,	6,	12,	20,	25,	36,	49,	49,	56,	56,	 63,	63,	   70,	  70,	 77],
                           [0,	3,	6,	12,	20,	30,	36,	49,	64,	64,	72,	 72,	80,	   80,	  88,	 96],
                           [0,	3,	8,	12,	20,	30,	42,	49,	64,	81,	81,	 90,	90,	   99,	  108,	 108],
                           [0,	3,	8,	15,	20,	30,	42,	56,	64,	81,	100, 100,	110,   120,	  120,	 130],
                           [0,	3,	8,	15,	20,	30,	42,	56,	64,	81,	100, 121,	121,   132,	  132,	 143],
                           [0,	3,	8,	15,	24,	30,	42,	56,	72,	81,	100, 121,	144,   144,	  156,	 156],
                           [0,	3,	8,	15,	24,	35,	42,	56,	72,	90,	100, 121,	144,   169,	  169,	 182],
                           [0,	3,	8,	15,	24,	35,	48, 56,	72,	90,	110, 132,	144,   169,	  196,	 196],
                           [0,	3,	8,	15,	24,	35,	48,	63,	72,	90,	110, 132,	144,   169,	  196,	 225]])


# lut_diff = lut_ideal_15 - lut_actual_15
lut_diff = lut_ideal_225 - lut_actual_225


class Net(nn.Module):
    def __init__(self, mnist=True):

        super(Net, self).__init__()
        if mnist:
            self.fc0 = nn.Linear(784, 800)
            self.fc1 = nn.Linear(800, 500)
            self.flatten_shape = 784
        else:
            self.fc1 = nn.Linear(1250, 500)
            self.flatten_shape = 1250

        self.fc2 = nn.Linear(500, 10)

    def forward(self, x, vis=False, axs=None):
        X = 0
        y = 0

        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


# # # Quantisation of Network

# # ## Quantisation Functions
QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    # Calc Scale and zero point of next
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    return scale, zero_point


def calcScaleZeroPointSym(min_val, max_val, num_bits=8):
    # Calc Scale
    max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.

    scale = max_val / qmax

    return scale, 0


def quantize_tensor(x, num_bits=8, min_val=None, max_val=None, verbose=False):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()

    if verbose:
        print('Quant   scale: ' + str(scale))

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


def quantize_tensor_sym(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.

    scale = max_val / qmax

    q_x = x / scale

    q_x.clamp_(-qmax, qmax).round_()
    q_x = q_x.round()
    return QTensor(tensor=q_x, scale=scale, zero_point=0)


def dequantize_tensor_sym(q_x):
    return q_x.scale * (q_x.tensor.float())


# # ## Get Stats for Quantising Activations of Network.
# This is done by running the network with around 1000 examples and
# getting the average min and max activation values before and after each layer.

# # Get Min and max of x tensor, and stores it
def updateStats(x, stats, key):
    max_val, _ = torch.max(x, dim=1)
    min_val, _ = torch.min(x, dim=1)

    # add ema calculation

    if key not in stats:
        stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
    else:
        stats[key]['max'] += max_val.sum().item()
        stats[key]['min'] += min_val.sum().item()
        stats[key]['total'] += 1

    weighting = 2.0 / (stats[key]['total']) + 1

    if 'ema_min' in stats[key]:
        stats[key]['ema_min'] = weighting * (min_val.mean().item()) + (1 - weighting) * stats[key]['ema_min']
    else:
        stats[key]['ema_min'] = weighting * (min_val.mean().item())

    if 'ema_max' in stats[key]:
        stats[key]['ema_max'] = weighting * (max_val.mean().item()) + (1 - weighting) * stats[key]['ema_max']
    else:
        stats[key]['ema_max'] = weighting * (max_val.mean().item())

    stats[key]['min_val'] = stats[key]['min'] / stats[key]['total']
    stats[key]['max_val'] = stats[key]['max'] / stats[key]['total']

    return stats


class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8, min_val=None, max_val=None, verbose=False):
        x = quantize_tensor(x, num_bits=num_bits, min_val=min_val, max_val=max_val, verbose=verbose)
        x_scale = x.scale
        x_q = x.tensor
        x = dequantize_tensor(x)
        return x, x_scale, x_q

    @staticmethod
    def backward(ctx, grad_output, grad_scale, grad_quantised):
        # straight through estimator
        return grad_output, None, None, None, None


def mapMultiplierModel(q_x, q_w):
    # x = quantize_tensor(x, num_bits)
    # w = quantize_tensor(w, num_bits)
    # m = x.scale * w.scale
    # x_quant = x.tensor
    q_w_t = torch.t(q_w)  # y = x.wT + b

    res = [[sum(lut_diff[a][b] for a, b in zip(X_row, Y_col)) for Y_col in zip(*q_w_t)] for X_row in q_x]
    res = torch.tensor(res)
    # res = torch.zeros([x_quant.size(0), w_quant_t.size(1)])
    # for i in range(x_quant.size(0)):
    #     for j in range(w_quant_t.size(1)):
    #         for k in range(w_quant_t.size(0)):
    #             # resulting matrix
    #             res[i][j] += lut_diff[x_quant[i][k]][w_quant_t[k][j]]

    # c = res*m
    return res


# ## Quantization Aware Training Forward Pass
def quantAwareTrainingForward(model, x, stats, vis=False, axs=None, sym=False, num_bits=8, act_quant=False,
                              verbose=False, test_quant=False):
    x = x.view(-1, 784)
    # print("\nfc0 input")
    x, m_x, q_x = FakeQuantOp.apply(x, num_bits, None, None, verbose)

    # FC0 LAYER
    fc0weight = model.fc0.weight.data
    # print("\nfc0 weight")
    model.fc0.weight.data, m_w, q_w = FakeQuantOp.apply(model.fc0.weight.data, num_bits, None, None, verbose)

    if act_quant or test_quant:
        c = mapMultiplierModel(q_x, q_w)

    x = model.fc0(x)

    if act_quant or test_quant:
        comp = m_x * m_w * c
        x = x - comp

    x = F.relu(x)

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc0')

    if act_quant:
        print("\nfc0 activation")
        x, m_x, q_x = FakeQuantOp.apply(x, num_bits, stats['fc0']['ema_min'], stats['fc0']['ema_max'], verbose)

    # FC1 LAYER
    fc1weight = model.fc1.weight.data
    # print("\nfc1 weight")
    model.fc1.weight.data, m_w, q_w = FakeQuantOp.apply(model.fc1.weight.data, num_bits, None, None, verbose)

    if act_quant or test_quant:
        c = mapMultiplierModel(q_x, q_w)

    x = model.fc1(x)

    if act_quant or test_quant:
        comp = m_x * m_w * c
        x = x - comp

    x = F.relu(x)

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc1')

    if act_quant:
        print("\nfc1 activation")
        x, m_x, q_x = FakeQuantOp.apply(x, num_bits, stats['fc1']['ema_min'], stats['fc1']['ema_max'], verbose)

    # FC2 LAYER
    fc2weight = model.fc2.weight.data
    # print("\nfc2 weight")
    model.fc2.weight.data, m_w, q_w = FakeQuantOp.apply(model.fc2.weight.data, num_bits, None, None, verbose)

    if act_quant or test_quant:
        c = mapMultiplierModel(q_x, q_w)

    x = model.fc2(x)

    if act_quant or test_quant:
        comp = m_x * m_w * c
        x = x - comp

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc2')

    if act_quant:
        print("\nfc2 activation")
        x, m_x, q_x = FakeQuantOp.apply(x, num_bits, stats['fc2']['ema_min'], stats['fc2']['ema_max'], verbose)

    return F.log_softmax(x, dim=1), fc0weight, fc1weight, fc2weight, stats


# # Train using Quantization Aware Training
def trainQuantAware(args, model, device, train_loader, optimizer, epoch, stats, act_quant=False, num_bits=4,
                    verbose=False):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, fc0weight, fc1weight, fc2weight, stats = quantAwareTrainingForward(model, data, stats,
                                                                        num_bits=num_bits,
                                                                        act_quant=act_quant,
                                                                        verbose=verbose,
                                                                        test_quant=False)

        model.fc0.weight.data = fc0weight
        model.fc1.weight.data = fc1weight
        model.fc2.weight.data = fc2weight

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args["log_interval"] == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCorrect: {}\tBatch: {}\tLength(data): {}\tAccuracy: {:.2f}'.format(
                    epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx+1) / len(train_loader), loss.item(), correct, batch_idx+1, len(data),
                           100. * correct / ((batch_idx+1) * len(data))))
    return stats


def testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=4):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, fc0weight, fc1weight, fc2weight, _ = quantAwareTrainingForward(model, data, stats,
                                                                        num_bits=num_bits,
                                                                        act_quant=act_quant,
                                                                        verbose=False,
                                                                        test_quant=False)

            model.fc0.weight.data = fc0weight
            model.fc1.weight.data = fc1weight
            model.fc2.weight.data = fc2weight

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def mainQuantAware(mnist=True):
    batch_size = 64
    test_batch_size = 64
    epochs = 2
    epochs_act_quant_active = 0
    num_bits = 4
    lr = 0.01
    momentum = 0.5
    seed = 1
    log_interval = 1
    save_model = True
    no_cuda = False
    verbose = False
    # act_quant = True

    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if mnist:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))  # Normalize (mean, stdev)
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # Normalize (mean, stdev)
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
        if epoch > epochs_act_quant_active:
            act_quant = True
            verbose = True
        else:
            act_quant = False
            verbose = False

        stats = trainQuantAware(args, model, device, train_loader, optimizer, epoch, stats, act_quant,
                                num_bits=num_bits, verbose=verbose)
        testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=num_bits)

    if (save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    return model, stats


torch.set_printoptions(threshold=1_000_000)
trained_model, old_stats = mainQuantAware()
