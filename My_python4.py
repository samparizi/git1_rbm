import urllib
import gzip
import numpy as np
import matplotlib.pyplot as plt


class RBM:
    def __init__(self, num_visible, num_hidden, epochs, k=2, lr=0.001, weight_decay=1e-4):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.v_bias = np.random.normal(-0.5, 0.05, num_visible)
        self.h_bias = np.random.normal(-0.2, 0.05, num_hidden)
        self.weights = np.random.normal(0, 0.05, [num_visible, num_hidden])
        self.weight_decay = weight_decay
        self.k = k
        self.lr = lr
        self.epochs = epochs

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

    def train(self, input_data, batch_size):
        errors = []
        costs = []
        input_data2 = input_data

        for i in xrange(self.epochs):
            input_data3 = np.random.permutation(input_data2)

            for j in xrange(300):
                input_data = input_data3
                # _id = np.random.choice(input_data[0], batch_size, replace=False)
                # input_data = input_data[_id, :]

                ff = j * batch_size
                ff2 = (j + 1) * batch_size
                input_data = input_data[ff:ff2, :]

                # print('input_data.shape', input_data.shape)

                # Positive phase
                pos_sample_h = self.sample_h(input_data)
                neg_sample_h = pos_sample_h

                # Negative phase
                for step in range(self.k):
                    neg_sample_v = self.sample_v(neg_sample_h)
                    neg_sample_h = self.sample_h(neg_sample_v)

                # Update and Save parameters
                pos_phase_h = np.dot(input_data.T, pos_sample_h)
                neg_phase_h = np.dot(neg_sample_v.T, neg_sample_h)

                self.weights += (pos_phase_h - neg_phase_h) * self.lr / batch_size
                # self.weights -= self.weights * self.weight_decay
                self.v_bias += np.sum(input_data - neg_sample_v) * self.lr / batch_size
                self.h_bias += np.sum(pos_sample_h - neg_sample_h) * self.lr / batch_size

            # Monitoring the training model
            # np.save('weight'+str(i), self.weights)
            # np.save('v_bias'+str(i), self.v_bias)
            # np.save('h_bias'+str(i), self.h_bias)

            error = np.sum((input_data - neg_sample_v) ** 2)
            cost = np.mean(self.free_energy(input_data)) - np.mean(self.free_energy(neg_sample_v))
            errors.append(error)
            costs.append(cost)
            print np.mean(self.weights), errors, costs

    # def a_rand_sample(self, input_data):
    #     # rand_sample = input_data[np.random.randint(input_data.shape[0]), :]
    #     rand_sample = np.random.randint(2, size=self.num_visible)
    #     fig_h, ax_h = plt.subplots(5, 4, sharex=True, sharey=True)
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     fig_v, ax_v = plt.subplots(5, 4, sharex=True, sharey=True)
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     ax_h = ax_h.flatten()
    #     ax_v = ax_v.flatten()
    #
    #     for i in xrange(self.epochs):
    #         img_h = self._sigmoid(
    #             np.dot(rand_sample, np.load(
    #  f               'weight' + str(i) + '.npy')) + np.load('h_bias' + str(i) + '.npy'))
    #
    #         img_v = self._sigmoid(
    #             np.dot(img_h, np.load(
    #                 'weight' + str(i) + '.npy').T) + np.load('v_bias' + str(i) + '.npy'))
    #         img_h = np.abs(img_h - 1)
    #         img_v = np.abs(img_v - 1)
    #
    #         ax_h[i].imshow(img_h.reshape(30, 30), cmap='Greys', interpolation='nearest')
    #         ax_v[i].imshow(img_v.reshape(28, 28), cmap='Greys', interpolation='nearest')
    #
    #     plt.show()

    def pre_sample_generating(self, v):
        return self.mean_v(self.mean_h(v))

    def sample_generating(self, sample_num):
        fig_v, ax_v = plt.subplots(8, 10, sharex=True, sharey=True)
        plt.xticks([])
        plt.yticks([])

        ax_v = ax_v.flatten()
        rand_sample = np.random.randint(2, size=[sample_num, self.num_visible])
        for m in xrange(40000):
            rand_sample = self.pre_sample_generating(rand_sample)
        img_v = rand_sample

        img_v = np.abs(img_v - 1)

        for i in xrange(sample_num):
            ax_v[i].imshow(img_v[i, :].reshape(28, 28), cmap='Greys', interpolation='nearest')

        plt.show()




    #def xxx:

    #     # input_data = np.random.normal(1, 0.05, [10000, self.num_visible])
    #     # print('input_data.shape', input_data.shape)
    #
    #     fig_vv, ax_vv = plt.subplots(5, 4, sharex=True, sharey=True)
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     ax_vv = ax_vv.flatten()
    #
    #     for i in xrange(20):
    #         # _id = np.random.choice(input_data[0], 1, replace=False)
    #         # input_data = input_data[_id, :]
    #         sample = np.random.normal(-0.1, np.random.randint(10), [1, 784])
    #         # rand_sample = input_data[np.random.randint(input_data.shape[0]), :]
    #         img_vv = np.abs(self.pre_sample_generating(sample) - 1)
    #         ax_vv[i].imshow(img_vv.reshape(28, 28), cmap='Greys', interpolation='nearest')
    #
    #     plt.show()

    # def sample_generating(self, input_data, batch_size):
    #     fig_vv, ax_vv = plt.subplots(5, 4, sharex=True, sharey=True)
    #
    #     plt.xticks([])
    #     plt.yticks([])
    #     ax_vv = ax_vv.flatten()
    #
    #     for i in range(20):
    #         # _id = np.random.choice(input_data[1], 700, replace=False)
    #         # input_data = input_data[_id, :]
    #
    #         self.weights = []
    #         self.v_bias = []
    #         self.h_bias = []
    #         self.v_bias = np.random.normal(-0.5, 0.05, self.num_visible)
    #         self.h_bias = np.random.normal(-0.2, 0.05, self.num_hidden)
    #         self.weights = np.random.normal(0, 0.05, [self.num_visible, self.num_hidden])
    #
    #         for j in range(self.epochs):
    #             _id = np.random.choice(input_data[0], batch_size, replace=False)
    #             input_data = input_data[_id, :]
    #
    #             # Positive phase
    #             pos_sample_h = self.sample_h(input_data)
    #
    #             # Negative phase
    #             for step in range(self.k):
    #                 neg_sample_v = self.sample_v(pos_sample_h)
    #                 neg_sample_h = self.sample_h(neg_sample_v)
    #
    #             # Update and Save parameters
    #             pos_phase_h = np.dot(input_data.T, pos_sample_h)
    #             neg_phase_h = np.dot(neg_sample_v.T, neg_sample_h)
    #             self.weights += (pos_phase_h - neg_phase_h) * self.lr / batch_size
    #             self.weights -= self.weights * self.weight_decay
    #             self.v_bias += np.sum(input_data - neg_sample_v) * self.lr / batch_size
    #             self.h_bias += np.sum(pos_sample_h - neg_sample_h) * self.lr / batch_size
    #             self.weights = self.weights
    #             self.v_bias = self.v_bias
    #             self.h_bias = self.h_bias
    #
    #         self.weights = self.weights
    #         self.v_bias = self.v_bias
    #         self.h_bias = self.h_bias
    #
    #         #sample = np.random.normal(-0.1, np.random.randint(10), [1, 784])
    #         sample = np.random.normal(0, 0.05, [1, 784])
    #         img_vv = np.abs(self.pre_sample_generating(sample) - 1)
    #
    #         ax_vv[i].imshow(img_vv.reshape(28, 28), cmap='Greys', interpolation='nearest')
    #
    #         self.weights = []
    #         self.v_bias = []
    #         self.h_bias = []
    #     plt.show()







# Loading the  MNIST database
img_size = 784
train_img = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
train_label = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
test_img = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
test_label = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

print('Downloading ..')

urllib.urlretrieve(train_img, 'train_img')
urllib.urlretrieve(train_label, 'train_label')
urllib.urlretrieve(test_img, 'test_img')
urllib.urlretrieve(test_label, 'test_label')

print('Converting ..')

with gzip.open('train_img', 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)
train_img = data.reshape(-1, img_size)/255.0
train_img = np.random.binomial(1, train_img)

with gzip.open('train_label', 'rb') as f:
    train_label = np.frombuffer(f.read(), np.uint8, offset=8)

with gzip.open('test_img', 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)
test_img = data.reshape(-1, img_size)/255.0
test_img = np.random.binomial(1, test_img)

with gzip.open('test_label', 'rb') as f:
    test_label = np.frombuffer(f.read(), np.uint8, offset=8)

print('Done')


# Training the RBM Model
print('Training the RBM Model..')

r = RBM(num_visible=784, num_hidden=900, epochs=15)
r.train(input_data=train_img, batch_size=200)

# Getting hidden features of a random sample
print('Getting hidden features of a random sample')
# r.a_rand_sample(input_data=test_img)

# Getting hidden features of a random sample
print('Getting hidden features of a random sample')
#r.a_rand_sample(input_data=test_img)

# Sample_generating
print('sample_generating')
r.sample_generating(sample_num=80)

