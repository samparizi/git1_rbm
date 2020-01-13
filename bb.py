import urllib
import gzip
import numpy as np
import matplotlib.pyplot as plt
import torch
#import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image



train_img = datasets.MNIST('../input_data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor()]))
for _, (train_img, target) in enumerate(train_img):
   train_img = torch.distributions.Bernoulli(Variable(train_img.view(-1, 784))).sample()

