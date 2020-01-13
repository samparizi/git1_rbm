import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid , save_image
import matplotlib.pyplot as plt
from math import log, exp




#load data
# transforms.Compose: Composes several transforms together.
#transforms.ToTensor: Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../My_RBM', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()])), batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../My_RBM', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])), batch_size=batch_size)

class RBM(nn.Module):
    def __init__(self, n_vis, n_hid, k):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
        self.k = k

    def vis_hid(self, v):
        p_h = F.sigmoid(F.linear(v, self.W, self.h_bias))
        sample_h = F.relu(torch.sign(p_h - Variable(torch.rand(p_h.size()))))
        return p_h, sample_h

    def hid_vis(self, h):
        p_v = F.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        sample_v = F.relu(torch.sign(p_v - Variable(torch.rand(p_v.size()))))
        return p_v, sample_v

    def forward(self, v):
        h0, h_ = self.vis_hid(v)
        for _ in range(self.k):
            v0_, v_ = self.hid_vis(h_)
            h0_, h_ = self.vis_hid(v_)
        return v, v_

    def free_energy(self, v):
        wx_b = F.linear(v, self.W, self.h_bias)
        vbias_term = v.mv(self.v_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()


rbm = RBM(n_vis=784, n_hid=500, k=1)
train_op = optim.SGD(rbm.parameters(), 0.1)

for epoch in range(4):
    loss_ = []
    for _, (data, target) in enumerate(train_loader):
        sample_data = Variable(data.view(-1, 784)).bernoulli()
        v, v1 = rbm(sample_data)
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        loss_.append(loss.data[0])
        train_op.zero_grad()
        loss.backward()
        train_op.step()

    print np.mean(loss_)


def monitoring(file_name,img):
    imgplot = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    plt.imshow(imgplot)
    plt.imsave(f, imgplot)


monitoring("original", make_grid(v.view(32, 1, 28, 28).data))
monitoring("generate", make_grid(v1.view(32, 1, 28, 28).data))













