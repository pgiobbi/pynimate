import pynimate
import numpy as np
import matplotlib.pyplot as plt

# Generate some fake data
niters = 1000
nobjs = 5
ndims = 2

data = np.random.randn(nobjs*ndims*niters).reshape(nobjs, ndims, niters)

fig, (ax1, ax2) = plt.subplots(1, 2)

# Just to show something
ax1.plot(data[:,0,:], data[:,1,:], alpha=0.5, lw=0.5)
ax1.set_title('Traces')
ax2.set_title('Animation')

# Plot! No hassles!
pynimate.anim(data, order='odi', interval=50, lag=20, traces=True,
              fig=fig, ax=ax2, scatter_kwargs=dict(alpha=0.8),
              xlim=ax1.get_xlim(), ylim=ax1.get_ylim(),
              traces_kwargs=dict(lw=0.4))
