import wget

print('Downloading MNIST raw data...')
wget.download('http://www-users.math.umn.edu/~jwcalder/MNIST_raw.npz','./Data/MNIST_raw.npz')
print('\n')

print('Downloading FashionMNIST raw data...')
wget.download('http://www-users.math.umn.edu/~jwcalder/FashionMNIST_raw.npz','./Data/FashionMNIST_raw.npz')
print('\n')

print('Downloading CIFAR raw data...')
wget.download('http://www-users.math.umn.edu/~jwcalder/cifar_raw.npz','./Data/cifar_raw.npz')
print('\n')
