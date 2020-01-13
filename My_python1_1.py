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
print('Done')

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
# done


class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_hidden = num_hidden
        self.num_visible = num_visible

        # Initialize a weight matrix
        # Ref: Xavier Glorot and Yoshua Bengio
        np_rng = np.random.RandomState(1234)
        self.weights = np.asarray(np_rng.uniform(
            low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            size=(num_visible, num_hidden)))
        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)

        v_bias = np.ones(num_visible)
        self.v_bias = v_bias
        h_bias = np.zeros(num_hidden)
        self.h_bias = h_bias

    def _linear(self, x, w, b):
        return x.dot(w) + b

    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def _bernoulli(self, s):
        return np.random.binomial(1, s)

    def mean_h(self, v):
        pre_sig_act = v.dot(self.weights) + self.v_bias
        return self._sigmoid(pre_sig_act)

    # def prop_up(self, v):
    #     pre_sig_act = F.linear(v, self.W, self.h_bias)
    #     p_h = F.sigmoid(pre_sig_act)
    #     return pre_sig_act, p_h
    #
    # def sample_h_v(self, v0_sample):
    #     pre_sig_h1, h1_mean = self.prop_up(v0_sample)
    #     h1_sample = F.relu(torch.sign(p_h - Variable(torch.rand(p_h.size()))))
    #     return pre_sig_h1, h1_mean, h1_sample

    def forward(self, v):
        pre_sig_act = v.dot(self.weights) + self.v_bias
        p_h = self._sigmoid(pre_sig_act)
        h1_sample = np.random.binomial(1, p_h)
        return pre_sig_act, p_h, h1_sample

    def backward(self, h):
        pre_sig_act = h.dot(self.weights.t()) + self.h_bias
        v_h = self._sigmoid(pre_sig_act)
        v1_sample = np.random.binomial(1, p_h)
        return pre_sig_act, v_h, v1_sample

    def train(self, data=train_img, max_epochs=1000, learning_rate=0.1):
        num_examples = data.shape[0]
        data = np.insert(data, 0, 1, axis=1)

        for epoch in range(max_epochs):
            pos_hidden_activations = np.dot(data, self.weights)
            pos_hidden_probs = self._sigmoid(pos_hidden_activations)
            pos_hidden_probs[:, 0] = 1  # Fix the bias unit.
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
            pos_associations = np.dot(data.T, pos_hidden_probs)
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self._sigmoid(neg_visible_activations)
            neg_visible_probs[:, 0] = 1  # Fix the bias unit.
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._sigmoid(neg_hidden_activations)
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
            self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)
            error = np.sum((data - neg_visible_probs) ** 2)

            if self.debug_print:
                print("Epoch %s: error is %s" % (epoch, error))

    def run_visible(self, data=train_img):
        num_examples = data.shape[0]
        hidden_states = np.ones((num_examples, self.num_hidden + 1))
        data = np.insert(data, 0, 1, axis=1)
        hidden_activations = np.dot(data, self.weights)
        hidden_probs = self._sigmoid(hidden_activations)
        hidden_states[:, :] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
        hidden_states = hidden_states[:, 1:]
        return hidden_states

    def run_hidden(self, data):
        num_examples = data.shape[0]
        visible_states = np.ones((num_examples, self.num_visible + 1))
        data = np.insert(data, 0, 1, axis=1)
        visible_activations = np.dot(data, self.weights.T)
        visible_probs = self._sigmoid(visible_activations)
        visible_states[:, :] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
        visible_states = visible_states[:, 1:]
        return visible_states

    def daydream(self, num_samples):
        samples = np.ones((num_samples, self.num_visible + 1))
        samples[0, 1:] = np.random.rand(self.num_visible)

        for i in range(1, num_samples):
            visible = samples[i - 1, :]
            hidden_activations = np.dot(visible, self.weights)
            hidden_probs = self._sigmoid(hidden_activations)
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            hidden_states[0] = 1
            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = self._sigmoid(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible + 1)
            samples[i, :] = visible_states
        return samples[:, 1:]




# if __name__ == '__main__':
#   r = RBM(num_visible = 6, num_hidden = 2)
#   training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
#   r.train(training_data, max_epochs = 5000)
#   print(r.weights)
#   user = np.array([[0,0,0,1,1,0]])
#   print(r.run_visible(user))


if __name__ == '__main__':
    r = RBM(num_visible=784, num_hidden=400)
    print(r.weights)
    #print("Epoch %s: error is %s" % (epoch, error))


# img = img
# label = train_label[1]
# print(label)






# print(img)
# print(img.shape)
# img = img.reshape(28, 28)
# print(img.shape)
#
#
# plt.imshow(img, cmap='gray')
# plt.show()
#
#
#
#
# # Show the sample image
# img = train_img[1]
# label = train_label[1]
# print(label)
#
# print(img)
# print(img.shape)
# img = img.reshape(28, 28)
# print(img.shape)
#
#
# plt.imshow(img, cmap='gray')
# plt.show()


