import urllib
import gzip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class RBM:
    def __init__(self, num_visible, num_hidden=1000, k=2, lr=0.001, weight_decay=1e-4):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.v_bias = np.zeros(num_visible)
        self.h_bias = np.zeros(num_hidden)
        self.h_samples = h_samples
        self.k = k
        self.lr = lr
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
        return self._sigmoid(v.dot(self.weights) + self.h_bias)

    def sample_h(self, v):
        return np.random.binomial(1, self.mean_h(v))

    def mean_v(self, h):
        return self._sigmoid(h.dot(self.weights.t()) + self.v_bias)

    def sample_v(self, h):
        return np.random.binomial(1, self.mean_v(h))

    def free_energy(self, v):
        linear_bias_term = v.dot(self.h_bias)
        pre_nonlinear_term = v.dot(self.weights) + self.v_bias
        nonlinear_term = np.sum(np.log1p(np.exp(pre_nonlinear_term)), axis=1)
        return -linear_bias_term - nonlinear_term

    def gibbs(self, v):
        return self.sample_v(self.sample_h(v))

    def train(self, input_data, max_epoch, batch_size):

        for i in xrange(max_epoch):
            _id = np.random.choice(input_data.shape(0), batch_size, replace=False)
            input_data= input_data[_id, :]

            # Positive phase
            pos_mean_h = self.mean_h(input_data)
            pos_sample_h = self.sample_h(input_data)

            # Negative phase
            for step in range(self.k):
                # neg_mean_v = self.mean_v(pos_sample_h)
                # neg_mean_h = self.mean_h(neg_mean_v)
                neg_sample_v = self.sample_v(pos_sample_h)
                neg_sample_h = self.sample_h(neg_sample_v)

            # Update parameters
            pos_phase_h = np.dot(input_data.T, pos_sample_h)
            neg_phase_h = np.dot(neg_sample_v.T, neg_sample_h)
            # batch_size = input_data.shape(0)
            self.weights += (pos_phase_h - neg_phase_h) * self.lr / batch_size
            self.weights -= self.weights * self.weight_decay
            self.v_bias += np.sum(input_data - neg_sample_v, axis=1) * self.lr / batch_size
            self.h_bias += np.sum(pos_sample_h - neg_sample_h, axis=1) * self.lr / batch_size

            error = np.sum((input_data - neg_sample_v) ** 2)
            cost = np.mean(self.free_energy(input_data)) - np.mean(self.free_energy(neg_sample_v))

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

print('Training the RBM Model..')

r = RBM(num_visible=784, num_hidden=1000)
r.train(input_data=train_img, max_epoch=5000, batch_size=64)
for epoch in range(10):
    error = 0.0
    cost = 0.0
    error, cost = r.train()
    error += error
    cost += cost
    print(error, cost)

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






