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
rng = numpy.random
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
    def __init__(self, n_vis=784, n_hid=500, w=None, v_bias=None, h_bias=None, input=None):
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
        self.input = input
        if not input:
            self.input = torch.Tensor('input', 2)

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

    def gibbs_vhv(self, v0_sample):
        pre_sig_h1, h1_mean, h1_sample = self.sample_h_v(v0_sample)
        pre_sig_v1, v1_mean, v1_sample = self.sample_v_h(h1_sample)
        return [pre_sig_h1, h1_mean, h1_sample,
                pre_sig_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        pre_sig_v1, v1_mean, v1_sample = self.sample_v_h(h0_sample)
        pre_sig_h1, h1_mean, h1_sample = self.sample_h_v(v1_sample)
        return [pre_sig_v1, v1_mean, v1_sample,
                pre_sig_h1, h1_mean, h1_sample]

    def free_energy(self, v_sample):
        wx_b = F.linear(v, self.W, self.h_bias)
        vbias_term = F.dot((v_sample, self.v_bias))
        hidden_term = torch.sum(torch.log1p((torch.exp(wx_b))), 1)
        return -hidden_term - vbias_term

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        pre_sig_ph, ph_mean, ph_sample = self.sample_h_v(self.input)
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        (
            [
                pre_sig_nvs,
                nv_means,
                nv_samples,
                pre_sig_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = 'theano.scan'(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k,
            name="gibbs_hvh"
        )
        chain_end = nv_samples[-1]
        cost = torch.mean(self.free_energy(self.input)) - torch.mean(self.free_energy(chain_end))
        gparams = 'T.grad'(cost, slef.params, consider_constant=[chain_end])
        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * 'T.cast'(
                lr,
                dtype=theano.config.floatX
            )
        if persistent:
            updates[persistent] = nh_samples[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            monitoring_cost = self.get_recontruction_cost(updates, pre_sig_nvs[-1])
        return monitoring_cost, updates



train_rbm = 'theano.function'(
    [index],
    cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index +1) * batch_size]
    },
    name='train_rbm'
)
plotting_time = 0.
start_time = timeit.default_timer()
for epoch in range(training_epochs):
    mean_cost = []
    for batch_index in range(n_train_batches):
        mean_cost += [train_rbm(batch_index)]

    print('Training epoch %d, cost is ' % epoch, np.mean(mean_cost))

    plotting_start = timeit.default_timer()
    image = Image.fromarray(
        tile_raster_images(
            X=torch.t(rbm.W.get_value(borrow=True)),
            img_shape=(28, 28),
            title_shape=(10, 10),
            title_spacing(1, 1)
        )
    )
    image.save('filters_at_epoch_%i.png' % epoch)
    plotting_stop = timeit.default_timer()
    plotting_time += (plotting_stop - plotting_start)

end_time = timeit.default_timer()
pretraining_time = (end_time - start_time) - plotting_time
print('Training took %f minutes' % (pretraining_time / 60.))

number_of_test_samples =test_set_x.get_value(borrow=True).shape[0]
test_idx = rng.randint(number_of_test_samples - n_chains)
persistent_vis_chain = 'theano.shared'(
    np.asarray(
        test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
    )
)


plot_every = 1000
(
    [
        presig_hids,
        hid_mfs,
        hid_samples,
        presig_vis,
        vis_mfs,
        vis_samples
    ],
    updates
) = 'theano.scan'(
    rbm.gibbs_vhv,
    outputs_info=[None, None, None, None, None, persistent_vis_chain],
    n_steps=plot_every,
    name="gibbs_vhv"
)

updates.update({persistent_vis_chain: vis_samples[-1]})

sample_fn = 'theano.function'(
    [],
    [
        vis_mfs[-1],
        vis_samples[-1]
    ],
    updates=updates,
    name='sample_fn'
)
image_data = np.zeros(
    (29 * n_samples + 1, 29 * n_chains - 1),
    dtype='uint8'
)

for idx in range(n_samples):
    vis_mf, vis_sample = sample_fn()
    print(' ... plotting sample %d' % idx)
    image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
        X=vis_mf,
        img_shape=(28, 28),
        tile_shape=(1, n_chains),
        tile_spacing=(1, 1)
    )

image = Image.fromarray(image_data)
image.save('samples.png')






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













