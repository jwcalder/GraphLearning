import wget
from os import path

if not path.isfile('./Data/MNIST_raw.npz'):
    print('Downloading MNIST raw data...')
    wget.download('http://www-users.math.umn.edu/~jwcalder/MNIST_raw.npz','./Data/MNIST_raw.npz')
    print('\n')


if not path.isfile('./Data/MNIST_vae.npz'):
    print('Downloading MNIST vae data...')
    wget.download('http://www-users.math.umn.edu/~jwcalder/MNIST_vae.npz','./Data/MNIST_vae.npz')
    print('\n')

if not path.isfile('./Data/FashionMNIST_raw.npz'):
    print('Downloading FashionMNIST raw data...')
    wget.download('http://www-users.math.umn.edu/~jwcalder/FashionMNIST_raw.npz','./Data/FashionMNIST_raw.npz')
    print('\n')

if not path.isfile('./Data/FashionMNIST_vae.npz'):
    print('Downloading FashionMNIST vae data...')
    wget.download('http://www-users.math.umn.edu/~jwcalder/FashionMNIST_vae.npz','./Data/FashionMNIST_vae.npz')
    print('\n')

#if not path.isfile('./Data/cifar_raw.npz'):
    #print('Downloading CIFAR raw data...')
    #wget.download('http://www-users.math.umn.edu/~jwcalder/cifar_raw.npz','./Data/cifar_raw.npz')
    #print('\n')

#if not path.isfile('./Data/cifar_aet.npz'):
    #print('Downloading CIFAR aet data...')
    #wget.download('http://www-users.math.umn.edu/~jwcalder/cifar_aet.npz','./Data/cifar_aet.npz')
    #print('\n')
