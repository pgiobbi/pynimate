import IO
import pynimate
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

plt.close('all')

dati2d=IO.readfile('samples2d.txt')
dati3d=IO.readfile('samples3d.txt')

# fig,(ax1,ax2) = plt.subplots(1,2,subplot_kw=dict(projection='3d'))

fig=plt.figure()
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122, projection='3d')


fig.suptitle('miao')

# Tight layout has to go before anim!
plt.tight_layout()
plt.subplots_adjust(top=0.88)

# pynimate.anim(dati2d, lag=20, interval=10, scatter=True, traces=True, traces_kwargs=dict(alpha=0.3), fig=fig, ax=ax1, labels=['a','b'])
#
ax1.plot(np.arange(10))
pynimate.anim(dati3d, lag=20, interval=10, scatter=True, traces=True, traces_kwargs=dict(alpha=0.3), fig=fig, ax=ax2, labels=['a','b','c'])
