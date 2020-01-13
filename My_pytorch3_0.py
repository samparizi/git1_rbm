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
import timeit
from PIL import Image
import tile_raster_images
import numpy.random as rng



batch_size = 64
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../My_RBM',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()])),
    batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../My_RBM',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor()])),
    batch_size=batch_size)


class RBM(nn.Module):
    def __init__(self, n_vis=784, n_hid=500, w=None, v_bias=None, h_bias=None, input_data=None):
        super(RBM, self).__init__()
        if w is None:
            w = nn.Parameter(torch.randn(n_hid, n_vis) * 1e-2)
        if v_bias is None:
            v_bias = nn.Parameter(torch.zeros(n_vis))
        if h_bias is None:
            h_bias = nn.Parameter(torch.zeros(n_hid))
        self.w = w
        self.v_bias = v_bias
        self.h_bias = h_bias
        self.input_data = input_data
        if not input_data:
            self.input_data = torch.Tensor('input_data', 2)

    def prop_up(self, v):
        pre_sig_act = F.linear(v, self.W, self.h_bias)
        p_h = F.sigmoid(pre_sig_act)
        return pre_sig_act, p_h

    def sample_h_v(self, v0_sample):
        pre_sig_h1, h1_mean = self.prop_up(v0_sample)
        h1_sample = F.relu(torch.sign(p_h - Variable(torch.rand(p_h.size()))))
        return pre_sig_h1, h1_mean, h1_sample

    def prop_down(self, h):
        pre_sig_act = F.linear(h, self.W.t(), self.v_bias)
        p_v = F.sigmoid(pre_sig_act)
        return pre_sig_act, p_v

    def sample_v_h(self, h0_sample):
        pre_sig_v1, v1_mean = self.prop_down(h0_sample)
        v1_sample = F.relu(torch.sign(p_v - Variable(torch.rand(p_v.size()))))
        return pre_sig_v1, v1_mean, v1_sample

    def gibbs_hvh(self, h0_sample):
        pre_sig_v1, v1_mean, v1_sample = self.sample_v_h(h0_sample)
        pre_sig_h1, h1_mean, h1_sample = self.sample_h_v(v1_sample)
        return [pre_sig_v1, v1_mean, v1_sample,
                pre_sig_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sig_h1, h1_mean, h1_sample = self.sample_h_v(v0_sample)
        pre_sig_v1, v1_mean, v1_sample = self.sample_v_h(h1_sample)
        return [pre_sig_h1, h1_mean, h1_sample,
                pre_sig_v1, v1_mean, v1_sample]

    def free_energy(self, v_sample):
        wx_b = F.linear(v, self.W, self.h_bias)
        vbias_term = F.dot((v_sample, self.v_bias))
        hidden_term = torch.sum(torch.log1p((torch.exp(wx_b))), 1)
        return -hidden_term - vbias_term

    def presistent_cd(self, k=5):
        self.k = k
        pre_sig_ph, ph_mean, ph_sample = self.sample_h_v(self.input_data)

        h0_sample = ph_sample
        for step in range(self.k):
            [pre_sig_v1, v1_mean, v1_sample,
             pre_sig_h1, h1_mean, h1_sample] = self.gibbs_hvh(h0_sample)
            [pre_sig_h1, h1_mean, h0_sample,
             pre_sig_v1, v1_mean, v1_sample] = self.gibbs_vhv(v1_sample)

        nv_sample = v1_sample[-1]
        cost = torch.mean(self.free_energy(self.input_data)) - torch.mean(self.free_energy(nv_sample))


rbm = RBM(n_vis=784, n_hid=500)
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













