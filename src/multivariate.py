import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

mu = np.array([0,0])
sigma = np.array([[1,0.8], [0.8,1]])

x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((x,y))
rv = multivariate_normal(mu, sigma)



plt.contourf(x, y, rv.pdf(pos), levels=20, cmap='viridis')
plt.title("2D Multivariate Gaussian")
plt.axis('equal')
plt.show()
