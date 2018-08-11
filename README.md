# pynimate
**Matplotlib animations. Made simple.**

Matplotlib is a powerful plotting tool that enables animations as well, but setting it is really painful. **pynimate**
does the annoying job for you and you can animate your data in 1, 2 or even 3 dimensions with a couple of lines of code with a friendly interface. 
Moreover there are a lot of convenient functions like animating points or traces, interpolation, saving the animation on file, 
automatic axis centering, animating on existing plots and more.
## Examples
In the following examples you will see several ways to use the pynimate module.

#### Example 1 - The easiest setup
```python
import pynimate
import numpy as np

# Generate points on a circle
theta = np.linspace(0, 2*np.pi, 100, endpoint=True)
x = np.cos(theta)
y = np.sin(theta)

pynimate.anim([x,y], order="di")  # The only line of code you to animate the data
```  
``order="di"`` stays for [dimension, iteration]. This means that the data array we are passing to ``pynimate.anim``
has the dimension (in our case, x or y) as the first index and the iteration as the second index. In fact ``[x,y]`` has a shape of ``(2, 100)``

The output is 

![Example 1 gif](https://media.giphy.com/media/2xPMR1hCmRwtvqE7wo/giphy.gif)


#### Example 2 - Many objects and some personalisation
You can easily animate more objects! You can decide to animate the points only or you can also animate the traces thanks to the ``lag`` parameter. 
You can personalize the color and whatever you like through the kwargs


```python
import pynimate
import numpy as np

# Generate 4 different random walks in 3 dimensions for 50 iterations
data = np.zeros((3, 4, 50))  # [dimension, object, iteration] -> order = "doi"
data[:,:,0] = np.random.randn(3,4)
for i in range(1, 50):
    data[:,:,i] = data[:,:,i-1] + np.random.randn(3,4) * 0.05

pynimate.anim(data, order="doi", interval=100, sigma=2, labels=["x","y","z"],
              lag=10, scatter_kwargs=dict(alpha=0.5, ms=3))
```
``sigma`` means that for each dimension the axis limit is given by (mean-sigma,mean+sigma), where mean 
is the average of all the positions in the given time frame

The output is 

![Example 2 gif](https://media.giphy.com/media/g04nn5KQRkiHDJ6ODD/giphy.gif)

#### Example 3 - Animations on many existing subplots
**pynimate** allows you to perform animation on several subplots! You can choose the dimensions to plot on each subplot through the ``plotdims`` parameter

```python
import pynimate
import numpy as np
import matplotlib.pyplot as plt

# Creating 4 subplots
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(5,5))

# 4 random walks in a 4-dimensional space
data = np.zeros((4, 4, 50))  # [dimension, object, iteration] -> order = "doi"
data[:,:,0] = np.random.randn(4,4)
for i in range(1, 50):
    data[:,:,i] = data[:,:,i-1] + np.random.randn(4,4) * 0.3

# Plotting projections of all the positions on each subplot
for i in range(4):
    ax1.plot(data[0,i,:], data[1,i,:], "o", color='k', alpha=0.3, ms=1)
    ax2.plot(data[2,i,:], data[1,i,:], "o", color='k', alpha=0.3, ms=1)
    ax3.plot(data[0,i,:], data[2,i,:], "o", color='k', alpha=0.3, ms=1)
    ax4.plot(data[3,i,:], data[2,i,:], "o", color='k', alpha=0.3, ms=1)

fig.set_tight_layout(True)

pynimate.anim(data, order="doi", interval=100, fig=fig, ax=(ax1,ax2,ax3,ax4),
              plotdims=[[0,1],[2,1],[0,2],[3,2]], interpolation=5, lag=10,
              scatter=False)
```
``plotdims`` will plot [0,1] components of each object on ax1 (0 on x-axis, 1 on y-axis), [2,1] on ax2 etc...


![Example 2 gif](https://media.giphy.com/media/aJ2t8l2xUH7bH3RI3Y/giphy.gif)

#### Example 4 - Interpolation
If your animation is too snappy you can use the ``interpolation`` parameter. ``interpolation - 1`` frames will be added between each couple of iteration points and everything will look smoother

Save the following data in ``motion.txt``
```txt
 -4.04  2.29 28.65
 -1.52  2.31 24.78
 -0.06  2.12 21.61
  0.79  2.14 18.96
  1.38  2.48 16.71
  1.95  3.19 14.84
  2.67  4.36 13.40
  3.69  6.15 12.50
  5.18  8.74 12.48
  7.29 12.18 13.99
 10.00 15.80 17.97
 12.77 17.55 24.85
 14.21 14.75 32.31
 13.07  8.22 35.77
  9.87  2.53 34.12
  6.32 -0.13 30.24
  3.61 -0.72 26.35
  1.94 -0.54 22.98
  1.03 -0.22 20.08
  0.60  0.06 17.57
  0.44  0.28 15.38
  0.42  0.50 13.47
  0.51  0.78 11.80
  0.70  1.17 10.36
  1.00  1.78  9.12
  1.50  2.73  8.12
  2.28  4.26  7.41
  3.54  6.66  7.22
  5.50 10.31  8.11
  8.42 15.29 11.28
 12.22 20.12 18.60
 15.58 20.16 29.95
 15.85 12.09 38.63
 12.18  2.26 38.46
  7.06 -2.70 33.43
  2.90 -3.82 28.50
  0.27 -3.72 24.68
 -1.27 -3.63 21.70
 -2.25 -3.94 19.31
 -3.07 -4.74 17.44
 -3.99 -6.09 16.16
 -5.21 -8.04 15.66
 -6.85 -10.58 16.34
 -8.91 -13.31 18.74
-11.08 -15.10 23.16
-12.59 -14.32 28.63
-12.54 -10.55 32.54
-10.76 -5.95 32.98
 -8.14 -2.82 30.74
 -5.73 -1.56 27.57
 ```

```python
import numpy as np
import pynimate

data = np.loadtxt("motion.txt")    # Shape is [50,3], so order="id"
pynimate.anim(dati, order="id", interval=200, interpolation=10, lag=5, 
              scatter=False)
```
Frame interpolation **off**           |  Frame interpolation **on** 
:-------------------------:|:-------------------------:
![Example 4 gif](https://media.giphy.com/media/5jZYfmDvozkR1oxKrf/giphy.gif) | ![Example 4 after interpolation gif](https://media.giphy.com/media/QmD8jyXkwF1nkkToHA/giphy.gif)
### Installation
The package is still in an early development stage, so you have to manually put the file into your python path.
