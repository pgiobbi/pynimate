from pynimate import anim
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d

# Generate some fake data
niters = 100
nobjs = 4
ndims = 5

data = np.random.randn(nobjs*ndims*niters).reshape(niters, nobjs, ndims)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')

plotdims = [[0, 1],     # on ax1
            [1, 2],     # on ax2
            [0, 4, 1]]  # on ax3

fig = anim(data, order="iod", colors=None, xlim=None, ylim=None, zlim=None,
           labels=None, interval=100, lag=10,
           plotdims=plotdims, flow=True, scatter=True,
           scatter_kwargs=dict(alpha=0.5, ms=3), traces=True,
           traces_kwargs=dict(alpha=0.5, lw=1), fig=fig,
           ax=(ax1, ax2, ax3), keeplims=False, save=False)
