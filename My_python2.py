import urllib
import gzip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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


class RBM:
    def __init__(self, num_visible, num_hidden, K, lr, weight_decay=1e-4):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.v_bias = np.zeros(num_visible)
        self.h_bias = np.zeros(num_hidden)
        self.h_samples = h_samples
        self.K = K
        self.lr =lr
        self.weight_decay = weight_decay


        # Initialize a weight matrix (Ref: Xavier Glorot and Yoshua Bengio)
        np_rng = np.random.RandomState(1234)
        self.weights = np.asarray(np_rng.uniform(
            low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            size=(num_visible, num_hidden)))

    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def mean_h(self, v):
        return self._sigmoid(v.dot(self.weights) + self.v_bias)

    def sample_h(self, v):
        return np.random.binomial(1, self.mean_h(v))

    def mean_v(self, h):
        return self._sigmoid(h.dot(self.weights.t()) + self.h_bias)

    def sample_v(self, h):
        return np.random.binomial(1, self.mean_v(h))

    def free_energy(self, v):
        linear_bias_term = v.dot(self.h_bias)
        pre_nonlinear_term = v.dot(self.weights) + self.v_bias
        nonlinear_term = np.sum(np.log1p(np.exp(pre_nonlinear_term)), axis=1)
        return -linear_bias_term - nonlinear_term

    def gibbs(self, v):
        return self.sample_v(self.sample_h(v))

    def train(self, input_data):

        # Positive phase
        pos_mean_h = self.mean_h(input_data)
        pos_sample_h = self.sample_h(input_data)
        pos_phase_h = np.dot(input_data.T, pos_sample_h)

        # Negative phase
        for step in range(self.K):
            neg_mean_v = self.mean_v(pos_sample_h)
            neg_mean_h = self.mean_h(neg_mean_v)
            # neg_sample_h = self.sample_h(neg_mean_v)

        neg_phase_h = np.dot(neg_mean_v.T, neg_mean_h)

        # Update parameters
        batch_size = input_data.shape(0)
        self.weights += (pos_phase_h - neg_phase_h) * self.lr / batch_size
        self.weights -= self.weights * self.weight_decay
        self.v_bias += np.sum(input_data - neg_mean_v, axis=1) * self.lr / batch_size
        self.h_bias += np.sum(pos_mean_h - neg_mean_h, axis=1) * self.lr / batch_size

        error = np.sum((input_data - neg_mean_v) ** 2)
        cost = np.mean(self.free_energy(pos_sample_h)) - np.mean(self.free_energy(neg_mean_v))

        return error, cost

    def sampling(self, num_samples):
        self.v_bias = np.ones(num_samples)
        self.h_bias = np.ones(self.num_hidden)
        samples = np.ones((num_samples, self.num_visible))
        samples[0, 1:] = np.random.rand(self.num_visible)

        for i in range(1, num_samples):
            sample_h = self.sample_h(samples[i - 1, :])
            samples[i, :] = self.sample_v(sample_h)
        return samples


r = RBM(num_visible=784, num_hidden=500)
r.train(input_data, max_epochs = 5000)
print(r.weights)

# Show a sample image
img = train_img[1]
label = train_label[1]
print(label)
print(img)
print(img.shape)
img = img.reshape(28, 28)
print(img.shape)
plt.imshow(img, cmap='gray')
plt.show()


