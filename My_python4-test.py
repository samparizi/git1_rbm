import urllib
import gzip
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image


class RBM:
    def __init__(self, num_visible, num_hidden, k=2, lr=0.001, weight_decay=1e-4):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.v_bias = np.random.normal(-0.5, 0.05, num_visible)
        self.h_bias = np.random.normal(-0.2, 0.05, num_hidden)
        self.k = k
        self.lr = lr
        self.weights = np.random.normal(0, 0.05, [num_visible, num_hidden])
        self.weight_decay = weight_decay

    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def mean_h(self, v):
        return self._sigmoid(np.dot(v, self.weights) + self.h_bias)

    def sample_h(self, v):
        return np.random.binomial(1, self.mean_h(v))

    def mean_v(self, h):
        return self._sigmoid(np.dot(h, self.weights.T) + self.v_bias)

    def sample_v(self, h):
        return np.random.binomial(1, self.mean_v(h))

    def free_energy(self, v):
        linear_bias_term = np.dot(v, self.v_bias)
        pre_nonlinear_term = np.dot(v, self.weights) + self.h_bias
        nonlinear_term = np.sum(np.log1p(np.exp(pre_nonlinear_term)))
        return -linear_bias_term - nonlinear_term

    def train(self, epochs, batch_size):
        ers = []
        cost = []

        for i in xrange(epochs):
            #_id = np.random.choice(input_data[0], batch_size, replace=False)
            #input_data = input_data[5, :]
            input_data = np.zeros([2, 784])
            # print(input_data.shape)
            # print('mean_h(input_data', self.mean_h(input_data).shape)

            linear_bias_term = np.dot(input_data, self.v_bias)
            # print('linear_bias_term.shape', linear_bias_term.shape)


            # Positive phase
            pos_sample_h = self.sample_h(input_data)
            # print('pos_sample_h.shape', pos_sample_h.shape)

            # Negative phase
            for step in range(self.k):
                neg_sample_v = self.sample_v(pos_sample_h)
                neg_sample_h = self.sample_h(neg_sample_v)

            # Update parameters
            pos_phase_h = np.dot(input_data.T, pos_sample_h)
            neg_phase_h = np.dot(neg_sample_v.T, neg_sample_h)
            self.weights += (pos_phase_h - neg_phase_h) * self.lr / batch_size
            self.weights -= self.weights * self.weight_decay
            # print('self.weights.shape', self.weights.shape)

            self.v_bias += np.sum(input_data - neg_sample_v) * self.lr / batch_size
            # print('self.v_bias.shape', self.v_bias.shape)
            # print('self.v_bias.shape', self.v_bias)


            self.h_bias += np.sum(pos_sample_h - neg_sample_h) * self.lr / batch_size
            # print('self.h_bias.shape', self.h_bias.shape)


            er = np.sum((input_data - neg_sample_v) ** 2)
            cst = np.mean(self.free_energy(input_data)) - np.mean(self.free_energy(neg_sample_v))
            ers.append(er)
            cost.append(cst)
            print np.mean(self.weights), ers, cost

        # s = test_img[np.random.randint(test_img.shape[0]), :]
        #
        # plt.imshow(s.reshape(28, 28), cmap='Greys')
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()

    # def testing(self):
    #
    #     s = test_img[np.random.randint(test_img.shape[0]), :]
    #
    #     plt.imshow(s.reshape(28, 28), cmap='Greys')
    #
    #     plt.show()
    #     l = 3
    #     W, bh, bv = [], [], []
    #     for i in xrange(l):
    #         W.append(np.load('dbn_params/W{0}.npy'.format(i)))
    #         bh.append(np.load('dbn_params/bh{0}.npy'.format(i)))
    #         bv.append(np.load('dbn_params/bv{0}.npy'.format(i)))
    #     for i in xrange(l):
    #         x = sample_vector(x, W[i], bh[i])
    #     fig, ax = plt.subplots(ncols=5, nrows=5, sharex=True, sharey=True)
    #     ax = ax.flatten()
    #     for i in xrange(25):
    #         img = np.copy(x)
    #         img[i] = np.abs(img[i] - 1)
    #         for k in reversed(xrange(l)):
    #             img = sample_vector(img, W[k].T, bv[k], bernoulli=False)
    #         ax[i].imshow(img.reshape(28, 28), cmap='Greys', interpolation='nearest')
    #     ax[0].set_xticks([])
    #     ax[0].set_yticks([])
    #     plt.show()




print('Training the RBM Model..')

r = RBM(num_visible=784, num_hidden=500)
r.train(epochs=3, batch_size=1)
# print(r.weights)

#for epoch in range(self.epochs):

    # print(r.error, r.cost)


# # Show a sample image
# img = train_img[1]
# # label = train_label[1]
# # print(label)
# # print(img)
# # print(img.shape)
# img = img.reshape(28, 28)
# # print(img.shape)
# plt.imshow(img, cmap='gray')
# plt.show()
