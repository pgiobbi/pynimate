"""
Created on Mon Jun 04 21:11:10 2018

@autor: pierm

Module that tries making data animation easy
Support for anims of several subplots coming soon..
"""

__version__= "1.1"

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def anim(data, order="iod", colors=None, xlim=None, ylim=None, zlim=None,
         labels=None, label_kwargs=None, interval=40, lag=1, scatter=True,
         scatter_kwargs=None,
         traces=True, traces_kwargs=None, fig=None, ax=None):
    """
    Animates 2d data using matplotlib.animation

    Parameters
    ----------
    data : array_like(1d/2d/3d)
        Data to animate

    order : string, optional
        What is the data format (i.e. what index correspond to what).
        There are three key-characters to let the procedure know your data:
            'i' : iteration
            'o' : object
            'd' : dimension
        The order of these characters tells the procedure how are data saved
        in ``data`` array.
        Example: "iod" means that data is rank=3 and is saved like data[i,o,d]
        NOTE: do thing below
            If data is 2d then it is assumed that data is order="id".

    colors : array_like[n_objects], optional
        Colors you wish to plot each object with. If None, colors are selected
        automatically. Colors need to be the same size of objects or bigger
        (falls back to None otherwise) or bigger.

    xlim : array_like[2]/tuple, optional
        Array with [xmin, xmax] to resize the figure boundaries. If None, set
        (xmin,xmax)= (xmean-3*std,xmean+3*std)

    ylim : array_like[2]/tuple, optional
        Array with [ymin, ymax]. Check xlim.

    zlim : array_like[2]/tuple, optional
        Array with [ymin, ymax]. Check xlim. If it is provided with ndims=3 then
        it is simply ignored

    labels : array_like[ndims], optional
        List/array of strings. A ValueError is raysed if len(labels)!=ndims

    label_kwargs : dict, optional
        Any extra keyword argument to send to the `set_(x/y/z)label` methods.

    interval : int, optional
        Interval in ms between frames

    lag : int, optional
        If 1 it makes a scatter animation. If lag=``some number`` then it lets
        the points stay on the figure for ``some number`` iteration after they
        are first plotted

    scatter : bool, optional
        Plot points if scatter is not None

    scatter_kwargs : dict, optional
        Any extra keyword arguments to send to matplotlib ``plot`` command for
        scatter points
        e.g. scatter_kwargs={'alpha':0.3} or scatter_kwargs=dict(alpha=0.3)

    traces : bool, optional
        Plot traces if lag is not None. At least one between scatter and traces
        needs to be True otherwise a ValueError is raised.

    traces_kwargs : dict, optional
        Any extra keywords arguments to send to matplotlib ``plot`` command for
        "line" points

    fig : matplotlib.Figure, optional
        Overplot onto the provided figure object.
        If None, create one.

    ax : matplotlib.axes, optional
        A axes instance on which to add the animation.
        If None, it creates a new subtplots instance. If it is not None but
        fig is, it overrides ax to None.

    Notes
    -----
    If you are planning to animate a subplot whose figure has more than one
    subplot, call anim after you plotted all the static subplots.
    """
    # Init dictionaries for kwargs


    if label_kwargs is None:
        label_kwargs = dict()
    if scatter_kwargs is None:
        scatter_kwargs = dict()
    if traces_kwargs is None:
        traces_kwargs  = dict()

    if not(scatter or traces):
        raise ValueError("At least one between scatter and traces should be "
                         "true! Don't you want to plot something?")

    if fig is None and ax is not None:
        raise ValueError("If you want to provide your own axis you need to "
                         "provide the figure the axis belogs to as well!")

    data=np.array(data)

    # Local functions used by anim only
    def init():  # Using lines as a global variable, it is okay
        for line in lines:
            line.set_data([],[])
        return lines

    def update_lines(index, data, lines):
        # Format for order!
        for j, line in enumerate(lines):
            firstindex=np.maximum(0,index-lag)  # Lag

            if ndims==2:
                # j%nobjs is to plot traces as well
                line.set_data(data[firstindex:index,j%nobjs,0],
                              data[firstindex:index,j%nobjs,1])
            else:
                # Traspose because the shape required is dim*points each dim
                # i.e. 3*10 if lag is 10 and ndims is 3
                line.set_data(data[firstindex:index,j%nobjs, 0:2].T)
                line.set_3d_properties(data[firstindex:index, j%nobjs, 2].T)
        return lines

    assert len(order)>=1 and len(order)<=3, "Size mismatch in order"

    # Checking if object input is valid
    iterindex = objindex = dimindex = None
    for i,character in enumerate(order):
        if character=='i':
            iterindex=i
        elif character=='o':
            objindex=i
        elif character=='d':
            dimindex=i
        else:
            raise ValueError("Wrong character in order")

    # Needs to be extended for partial input
    if np.any([iterindex, objindex, dimindex])==None:
        raise ValueError("Data not understood. Check order")

    niter=data.shape[iterindex]
    nobjs=data.shape[objindex]
    ndims=data.shape[dimindex]

    if ndims<2 or ndims>3:
        raise ValueError("Wrong dimension ndims = %i for anim"%ndims)

    if labels is not None:
        if len(labels)!=ndims:
            raise ValueError("len(labels) = %i does not match ndims = %i"
                             %(len(labels), ndims))

    # Swap axes to get the right layout
    if order!="iod":
        # Put iteration column in the first column
        if iterindex!=0:
            print data.shape
            if objindex==0:
                data=np.swapaxes(data, iterindex,0)
                objindex=iterindex
                iterindex=0
            else:
                # If objects weren't in the first place, then dims were!
                data=np.swapaxes(data, iterindex,0)
                dimindex=iterindex
                iterindex=0

        # If objects are not in the second column, swap 2nd and 3rd
        if objindex!=1:
            data=np.swapaxes(data, 1,2)

    # Create a new figure if one wasn't provided
    if fig is None:
        if ndims==2:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    else:
        # If fig is provided but not ax, plot over existing ax if possible
        if ax is None:
            ax=fig.axes
            if len(ax)==1:
                ax=fig.axes[0]  # matplotlib return a list of axes even if
                                # n_axes=1, but we need the axis, so [0]
            else:
                raise ValueError("Provided figure has %i axes, but we "
                                 "need one axis only!"%len(ax))

    # At this point if ax was provided, it is used automatically

    # ====== Setting plot limits ======
    if xlim is None:
        xmean=np.mean(data[:,:,0])
        xstd=np.std(data[:,:,0])
        xlim = xmean-3*xstd, xmean+3*xstd
    try:
        if ndims==2:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim3d(xlim)
    except:
        raise ValueError("xlim not understood")

    if ylim is None:
        ymean=np.mean(data[:,:,1])
        ystd=np.std(data[:,:,1])
        ylim = ymean-3*ystd, ymean+3*ystd
    try:
        if ndims==2:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim3d(ylim)
    except:
        raise ValueError("ylim not understood")

    if ndims==3:
        zmean=np.mean(data[:,:,2])
        zstd=np.std(data[:,:,2])
        zlim = zmean-3*zstd, zmean+3*zstd
        try:
            ax.set_zlim3d(zlim)
        except:
            raise ValueError("zlim not understood")
    # ====== Plot limits set ======

    # Setting labels
    if labels is not None:
        ax.set_xlabel(labels[0], **label_kwargs)
        ax.set_ylabel(labels[1], **label_kwargs)
        if ndims==3:
            ax.set_zlabel(labels[2], **label_kwargs)

    # If colors are not provided, let matplotlib decide and use default ones
    if colors is None:
        colors=[None for i in range(nobjs)]

    # Create scatter `lines` only if the user wants to plot points
    if scatter:
        if ndims==2:
            lines=[ax.plot([],[], 'o', color=colors[i],**scatter_kwargs)[0]
                   for i in range(nobjs)]
        else:
            lines = [ax.plot(data[0:1, i, 0], data[0:1, i,1],
                     data[0:1, i,2], 'o', color=colors[i],
                     **scatter_kwargs)[0] for i in range(nobjs)]
    else:
        lines=[]

    # Add lines for traces
    if traces:
        for i in range(nobjs):
            if ndims==2:
                lines.append(ax.plot([],[], color=colors[i],
                             **traces_kwargs)[0])
            else:
                lines.append(ax.plot(data[0:1, i, 0], data[0:1, i,1],
                         data[0:1, i,2], color=colors[i], **traces_kwargs)[0])

    # Finally animate
    if ndims==2:
        anim=animation.FuncAnimation(fig, update_lines, init_func=init,
             interval=interval, fargs=(data, lines), frames=niter, blit=True)
    else:
        anim=animation.FuncAnimation(fig, update_lines, interval=interval,
              fargs=(data, lines), frames=niter, blit=True)

    plt.show()
