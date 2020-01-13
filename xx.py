
import urllib
import gzip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#Load the MNIST dataset
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

with gzip.open('train_label', 'rb') as f:
    train_label = np.frombuffer(f.read(), np.uint8, offset=8)

with gzip.open('test_img', 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)
test_img = data.reshape(-1, img_size)/255.0

with gzip.open('test_label', 'rb') as f:
    test_label = np.frombuffer(f.read(), np.uint8, offset=8)

print('Done')


# Show the sample image
img = train_img[1]
label = train_label[1]
print(label)

print(img)
print(img.shape)
img = img.reshape(28, 28)
print(img.shape)


plt.imshow(img, cmap='gray')
plt.show()
