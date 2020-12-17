import urllib.request
from os import path

if not path.isfile('./Data/MNIST_raw.npz'):
    print('Downloading MNIST raw data...')
    urllib.request.urlretrieve('http://www-users.math.umn.edu/~jwcalder/MNIST_raw.npz', './Data/MNIST_raw.npz')

if not path.isfile('./Data/MNIST_vae.npz'):
    print('Downloading MNIST vae data...')
    urllib.request.urlretrieve('http://www-users.math.umn.edu/~jwcalder/MNIST_vae.npz','./Data/MNIST_vae.npz')

if not path.isfile('./Data/FashionMNIST_raw.npz'):
    print('Downloading FashionMNIST raw data...')
    urllib.request.urlretrieve('http://www-users.math.umn.edu/~jwcalder/FashionMNIST_raw.npz','./Data/FashionMNIST_raw.npz')

if not path.isfile('./Data/FashionMNIST_vae.npz'):
    print('Downloading FashionMNIST vae data...')
    urllib.request.urlretrieve('http://www-users.math.umn.edu/~jwcalder/FashionMNIST_vae.npz','./Data/FashionMNIST_vae.npz')

#if not path.isfile('./Data/cifar_raw.npz'):
    #print('Downloading CIFAR raw data...')
    #urllib.request.urlretrieve('http://www-users.math.umn.edu/~jwcalder/cifar_raw.npz','./Data/cifar_raw.npz')

#if not path.isfile('./Data/cifar_aet.npz'):
    #print('Downloading CIFAR aet data...')
    #urllib.request.urlretrieve('http://www-users.math.umn.edu/~jwcalder/cifar_aet.npz','./Data/cifar_aet.npz')
