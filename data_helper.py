import numpy as np
from torchvision import datasets, transforms
def convert_to_one_hot(arr):
    num_samples = arr.shape[0]
    one_hot = np.zeros((num_samples, 10))
    one_hot[np.arange(num_samples), arr] = 1
    return one_hot

def load_data():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for img, label in train_dataset:
        x_train.append(img.numpy())
        y_train.append(label)
    for img, label in test_dataset:
        x_test.append(img.numpy())
        y_test.append(label)
    x_train = np.array(x_train).reshape(60000,784)
    y_train = np.array(y_train)
    x_test = np.array(x_test).reshape(len(x_test), 784)
    y_test = np.array(y_test)
    y_train = convert_to_one_hot(y_train)
    y_test = convert_to_one_hot(y_test)


    return x_train, y_train, x_test, y_test

