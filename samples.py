
import random
import numpy as np
from numpy.random import randn

def generate_real_samples(dataset, n_samples):
	# choose random images
	ix = np.random.randint(0, dataset.shape[0], n_samples)
	# select the random images and assign it to X
	X = dataset[ix]
	# generate class labels and assign to y
	y = np.ones((n_samples, 1)) ##Label=1 indicating they are real
	return X, y


def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = np.random.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input


def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict using generator to generate fake samples. 
	X = generator.predict(x_input)
	# Class labels will be 0 as these samples are fake. 
	y = np.zeros((n_samples, 1))  #Label=0 indicating they are fake
	return X, y