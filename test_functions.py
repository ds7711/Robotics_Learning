import numpy as np
import robotics_library as rll
import matplotlib.pylab as plt


# test the implementation of the keras model
num_samples = 10000
num_dim = 2
x_scale = 10
noise_scale = 10
theta = np.random.randint(1, 10, size=(num_dim))

x = np.random.random((num_samples, num_dim)) * x_scale
y = np.dot(x, theta) + np.random.random(size=(num_samples)) * noise_scale

labels = np.random.random(10000)

nnn = rll.get_q_func([num_dim, 5, 1])

nnn.fit(x, y, batch_size=100, epochs=10)

y_hat = nnn.predict(x)

plt.scatter(y, y_hat)