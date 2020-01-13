import urllib
import gzip
import numpy as np
import matplotlib.pyplot as plt


class RBM:
    def __init__(self, num_visible, num_hidden, epochs, k, lr):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.v_bias = np.random.normal(-0.5, 0.05, num_visible)
        self.h_bias = np.random.normal(-0.2, 0.05, num_hidden)
        self.weights = np.random.normal(0, 0.05, [num_visible, num_hidden])
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

    def train(self, input_data, batch_size):
        errors = []

        for i in xrange(self.epochs):
            input_data2 = np.random.permutation(input_data)

            for j in xrange(300):

                batch_init = j * batch_size
                batch_end = (j + 1) * batch_size
                input_data = input_data2[batch_init:batch_end, :]

                # Positive phase
                pos_sample_h = self.sample_h(input_data)
                neg_sample_h = pos_sample_h

                # Negative phase
                for step in range(self.k):
                    neg_sample_v = self.sample_v(neg_sample_h)
                    neg_sample_h = self.sample_h(neg_sample_v)

                # Update parameters
                pos_phase_h = np.dot(input_data.T, pos_sample_h)
                neg_phase_h = np.dot(neg_sample_v.T, neg_sample_h)

                self.weights += (pos_phase_h - neg_phase_h) * self.lr / batch_size
                self.v_bias += np.sum(input_data - neg_sample_v) * self.lr / batch_size
                self.h_bias += np.sum(pos_sample_h - neg_sample_h) * self.lr / batch_size

            error = np.sum((input_data - neg_sample_v) ** 2)
            errors.append(error)
            print np.mean(self.weights), errors

    def rand_sampling(self, sample_num):
        fig_v, ax_v = plt.subplots(5, 8, sharex=True, sharey=True)
        ax_v = ax_v.flatten()
        plt.xticks([])
        plt.yticks([])

        rand_sample = np.random.randint(2, size=[sample_num, self.num_visible])

        for m in xrange(10000):
            rand_sample = self.sample_v(self.sample_h(rand_sample))
        img_v = np.abs(rand_sample - 1)

        for i in xrange(sample_num):
            ax_v[i].imshow(img_v[i, :].reshape(28, 28), cmap='Greys', interpolation='nearest')

        plt.show()


# Loading the  MNIST database
print('Downloading and Converting ..')

urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'train_img')
urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'train_label')
urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 'test_img')
urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 'test_label')
img_size = 784

with gzip.open('train_img', 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)
train_img = data.reshape(-1, img_size) / 255.0
train_img = np.random.binomial(1, train_img)

with gzip.open('train_label', 'rb') as f:
    train_label = np.frombuffer(f.read(), np.uint8, offset=8)

with gzip.open('test_img', 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)
test_img = data.reshape(-1, img_size) / 255.0
test_img = np.random.binomial(1, test_img)

with gzip.open('test_label', 'rb') as f:
    test_label = np.frombuffer(f.read(), np.uint8, offset=8)


# Training the RBM Model
print('Training the RBM Model..')

r = RBM(num_visible=784, num_hidden=1000, epochs=10, k=2, lr=0.001)
r.train(input_data=train_img, batch_size=200)

# Sample_generating
print('Random sample generating')
r.rand_sampling(sample_num=40)

