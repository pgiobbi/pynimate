"""
Created on Mon Jun 04 21:11:10 2018

@autor: pierm

Module that tries making data animation easy
Support for anims of several subplots coming soon..
"""

__version__ = "2.0"

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import warnings
import sys


def anim(data, order="iod", colors=None, xlim=None, ylim=None, zlim=None,
         labels=None, label_kwargs=None, interval=40, lag=1, fade=False,
         plotdims=None, flow=True, scatter=True, scatter_kwargs=None,
         traces=True, traces_kwargs=None, fig=None, ax=None, save=False,
         savename='animation.mp4'):
    """
    Animates 1d/2d/3d data using matplotlib.animation

    Parameters
    ----------
    data : array_like(1d/2d/3d)
        Data to animate
        # TODO : now it works only with rank=3, but it shouldn't be difficult.

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

    colors : array_like[n_objects]/string, optional
        Global colors for both scatterplot and traces.
        Colors you wish to plot each object with. If None, colors are selected
        automatically. Colors need to be the same size of objects or bigger
        (raises a ValueError otherwise). If only a single color is provided,
        make it a global color for all objects.
        e.g. colors=['r','g'] or colors='rg'
        TODO : fix colors when both scatter and traces are true

    xlim : array_like[2]/tuple, optional
        Array with [xmin, xmax] to resize the figure boundaries. If None, set
        (xmin,xmax)= (xmean-3*std,xmean+3*std)

    ylim : array_like[2]/tuple, optional
        Array with [ymin, ymax]. Check xlim.

    zlim : array_like[2]/tuple, optional
        Array with [ymin, ymax]. Check xlim. If it is provided with ndims=3
        then it is simply ignored

    labels : array_like[ndims](string), optional
        List/array of strings. A ValueError is raised if len(labels)<ndims.
        If the provided labels are more than ndims, labels are trimmed.

    label_kwargs : dict, optional
        Any extra keyword argument to send to the `set_(x/y/z)label` methods.

    interval : int, optional
        Interval in ms between frames

    lag : int/np.inf, optional
        If 1 it makes a scatter animation. If lag=``some number`` then it lets
        the points stay on the figure for ``some number`` iteration after they
        are first plotted. If lag==np.inf each point/line is going to persist
        on the figure.

    fade : bool, optional
        TODO with LineCollection
        Make the data slowly disappear by lowering `alpha` (if lag>1)

    plotdims : array_like(1,2 or 3)(ints), optional
        The indices of the params you wish to plot (order counts, so the first
        one will be on the x axis , the second on the y axis  etc.).
        If None, default is [0,1,2] etc.
        If len(plotdims)<0 or > 3 a ValueError is raised.
        If len(plotdims)>ndims, plotdims is trimmed -> plotdims[0:ndims]
        If plotdims contains two equal values, a ValueError is raised
        Example: plotdims=[0,2,1] plots param 0 on x, param 2 on y, param 1 on
        z

    flow : bool, optional
        Animation control for ndims=1.
        If True, data is plotted in 2d (where the extra dimension is time).
        If False, animation is in 1d (less cool :( )

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

    save : bool, optional
        Set True if you want to save the animation on file.

    savename : string, optional
        Name of the rendered file

    # TODO: Add more arguments to anim (maybe like a dictionary)


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
        traces_kwargs = dict()

    # ========= Start of error handling =========
    if not(scatter or traces):
        raise ValueError("At least one between scatter and traces should be "
                         "true! Don't you want to plot something?")

    if fig is None and ax is not None:
        raise ValueError("If you want to provide your own axis you need to "
                         "provide the figure the axis belogs to as well!")

    data = np.array(data)

    # TODO : Not yet implemented if len(order)!=3
    # Data rank must be 3
    assert len(order) >= 1 and len(order) <= 3, "Size mismatch in 'order' " \
                                                "argument"

    # Find the format of the provided data, getting various indices
    iterindex = objindex = dimindex = None
    for i, character in enumerate(order):
        if character == 'i':
            iterindex = i
        elif character == 'o':
            objindex = i
        elif character == 'd':
            dimindex = i
        else:
            raise ValueError("Wrong character in `order`")
        # TODO : handle silly inputs 'iii', 'oiooooo' etc

    # TODO : Needs to be extended for partial input (same as two above)
    if np.any(np.array([iterindex, objindex, dimindex]) is None):
        raise ValueError("Data not understood. Check `order` argument")

    niter = data.shape[iterindex]
    nobjs = data.shape[objindex]
    ndims = data.shape[dimindex]

    if ndims < 1:   # Is it reasonable? Could ndims be < 1? Whatever
        raise ValueError("Wrong dimension ndims = %i for anim" % ndims)

    if plotdims is None:
        # ndims = 1, 2 or 3 works even if plotdims is None since there is no
        # conflict between axes ...
        if ndims > 3:
            # but if ndims>3 axes are ambiguous!
            raise ValueError("Plot for ndims > 3 is allowed, but you need to "
                             "provide the params you want to plot through "
                             "`plotdims` argument!")
    else:
        plotdims = np.atleast_1d(plotdims)  # The line below wouldn't work!
        if np.any(plotdims > (ndims - 1)) or np.any(plotdims < 0):
            raise ValueError("'plotdims' is asking for a dimension that does "
                             "not exist!")
        if len(plotdims) != len(np.unique(plotdims)):
            raise ValueError("Value repetition in plotdims is not allowed")

        # plotdims overrides everything. The lines below work like a charm:
        #   - if len(plotdims)>ndims, plotdims is trimmed
        #   - if len(plotdims)<ndims, ndims is changed to len(plotdims) ...
        plotdims = plotdims[:ndims]
        ndims_old = ndims   # Saving old dimension for later use
        ndims = len(plotdims)
        if ndims < 1 or ndims > 3:
            # but you still need to plot 1d, 2d or 3d animations!
            raise ValueError("You can't animate less than 1d or more than 3d!")

    if labels is not None:
        if len(labels) < ndims:
            raise ValueError("You did't provide enough labels for the axes!"
                             "n_labels=%i, while ndims = %i"
                             % (len(labels), ndims))

    # If the user wants to, make points persist
    if lag == np.inf:
        lag = niter
    else:
        if lag <= 0:
            raise ValueError("lag should be positive!")

    # ========== End of error handling ==========

    # =========== Animation functions ===========
    # Local functions used by `anim` only
    def init():  # Using lines as a global variable, it is okay
        for line in lines:
            line.set_data([], [])
        return lines

    def update_lines(index, data, lines, lag, nobjs):
        for j, line in enumerate(lines):
            firstindex = np.maximum(0, index - lag)  # Lag

            if ndims == 1:
                ydata = np.zeros(np.minimum(index, lag))
                if flow:
                    # Flowing plot
                    ydata = np.arange(len(ydata), dtype=float)[::-1]/(lag - 1)
                    # ydata=np.arange(len(ydata))[::-1]/(len(ydata)-1)
                line.set_data(data[firstindex:index, j % nobjs, 0],
                              ydata)

            elif ndims == 2:
                # j%nobjs is to plot traces as well
                line.set_data(data[firstindex:index, j % nobjs, 0],
                              data[firstindex:index, j % nobjs, 1])
            else:
                # Traspose because the shape required is dim*points each dim
                # i.e. 3*10 if lag is 10 and ndims is 3
                line.set_data(data[firstindex:index, j % nobjs, 0:2].T)
                line.set_3d_properties(data[firstindex:index, j % nobjs, 2].T)
        return lines
    # ===========================================

    # Swap axes to get the right layout (iter,obj,dim)
    if order != "iod":
        # Put iteration column in the first column
        if iterindex != 0:
            if objindex == 0:
                data = np.swapaxes(data, iterindex, 0)
                objindex = iterindex
                iterindex = 0
            else:
                # If objects weren't in the first place, then dims were!
                data = np.swapaxes(data, iterindex, 0)
                dimindex = iterindex
                iterindex = 0

        # If objects are not in the second column, swap 2nd and 3rd
        if objindex != 1:
            data = np.swapaxes(data, 1, 2)

    # Masking data to plot only the required dimensions. It is pretty fast
    # This is safe since the axes have been swapped already
    if plotdims is not None:
        mask = np.zeros(ndims_old, dtype=bool)
        mask[plotdims] = True       # cool!
        data = data[:, :, mask]     # Only data with mask==True is kept

        # Swap dimensions to match plotdims (bubblesort)
        for i in range(len(plotdims)-1):
            for j in range(len(plotdims)-1-i):
                if plotdims[j] > plotdims[j+1]:
                    temparr, temp = np.copy(data[:, :, j+1]), \
                                    np.copy(plotdims[j+1])
                    data[:, :, j+1], plotdims[j+1] = data[:, :, j], plotdims[j]
                    data[:, :, j], plotdims[j] = temparr, temp
    # Create a new figure if one wasn't provided
    if fig is None:
        if ndims == 1 or ndims == 2:
            fig, ax = plt.subplots()
        elif ndims == 3:
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        else:
            # Error Handling was done already, but you never know...
            raise SystemExit(0)
    else:
        # If fig is provided but not ax, plot over existing ax if possible
        if ax is None:
            ax = fig.axes
            if len(ax) == 1:
                ax = fig.axes[0]    # matplotlib return a list of axes even if
                                    # n_axes = 1, but we need the axis, so [0]
            else:
                raise ValueError("Provided figure has %i axes, but we "
                                 "need one axis only!" % len(ax))

    # At this point if ax was provided, it is used automatically

    # ====== Setting plot limits ======
    # if limits are not provided, plot [mean +/- 3*sigma] for each component
    if xlim is None:
        xmean = np.mean(data[:, :, 0])
        xstd = np.std(data[:, :, 0])
        xlim = xmean-3*xstd, xmean+3*xstd

    if ndims == 1 or ndims == 2:
        ax.set_xlim(xlim)
    elif ndims == 3:
        ax.set_xlim3d(xlim)
    else:
        raise ValueError("'xlim' not understood. (Are you passing an axis "
                         "of the wrong type?)")

    if ylim is None:
        if ndims == 2 or ndims == 3:
            ymean = np.mean(data[:, :, 1])
            ystd = np.std(data[:, :, 1])
            ylim = ymean-3*ystd, ymean+3*ystd

    if ndims == 1:
        ax.set_ylim(-0.5, 1.5)
    elif ndims == 2:
        ax.set_ylim(ylim)
    elif ndims == 3:
        ax.set_ylim3d(ylim)
    else:
        raise ValueError("'ylim' not understood")

    if ndims == 3:
        zmean = np.mean(data[:, :, 2])
        zstd = np.std(data[:, :, 2])
        zlim = zmean-3*zstd, zmean+3*zstd

        try:
            ax.set_zlim3d(zlim)
        except ValueError:
            print("`zlim` not understood")
            sys.exit(1)
    # ====== Plot limits set ======

    # Setting labels
    if labels is not None:
        ax.set_xlabel(labels[0], **label_kwargs)
        if ndims == 2 or ndims == 3:
            ax.set_ylabel(labels[1], **label_kwargs)
        if ndims == 3:
            ax.set_zlabel(labels[2], **label_kwargs)

    # If colors are not provided, let matplotlib decide and use default ones
    if colors is None:
        colors = [None for i in range(nobjs)]
    else:
        # if provided colors are not enough, set the first one as a global one
        if len(colors) < nobjs:
            if len(colors) == 1:
                colors = [colors for i in range(nobjs)]
            else:
                raise ValueError("Size mismatch for colors: expected %i, "
                                 "got %i" % (nobjs, len(colors)))

    # Create scatter `lines` only if the user wants to plot points
    if scatter:
        if ndims == 1 or ndims == 2:
            lines = [ax.plot([], [], 'o', color=colors[i], **scatter_kwargs)[0]
                     for i in range(nobjs)]
        else:
            lines = [ax.plot(data[0:1, i, 0], data[0:1, i, 1],
                     data[0:1, i, 2], 'o', color=colors[i],
                     **scatter_kwargs)[0] for i in range(nobjs)]

        # Saving automatic colors to pass to traces
        for i in range(nobjs):
            colors[i] = lines[i].get_color()
    else:
        lines = []

    # Add lines for traces
    if traces:
        for i in range(nobjs):

            if ndims == 1 or ndims == 2:
                lines.append(ax.plot([], [], color=colors[i],
                             **traces_kwargs)[0])
            else:
                lines.append(ax.plot(data[0:1, i, 0], data[0:1, i, 1],
                             data[0:1, i, 2], color=colors[i],
                             **traces_kwargs)[0])

    def show_anim():
        # Just a quick fix using global variables to avoid nasty loops later
        global ln_an
        if ndims == 1 or ndims == 2:
            ln_an = animation.FuncAnimation(fig, update_lines, init_func=init,
                                            fargs=(data, lines, lag, nobjs),
                                            interval=interval, frames=niter,
                                            blit=True)
        else:
            ln_an = animation.FuncAnimation(fig, update_lines,
                                            fargs=(data, lines, lag, nobjs),
                                            interval=interval, frames=niter,
                                            blit=True)

    if not save:
        # Finally animate
        show_anim()
        plt.show()
    else:
        # Save animation
        fps = 1000/interval
        try:
            import subprocess   # Try to import this for really fast rendering
        except ImportError:
            print warnings.warn("import subprocess failed. Rendering will be "
                                " slower", ImportWarning)
            show_anim()
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, bitrate=180,
                            metadata=dict(artist='pynimate by pjoe95'))
            ln_an.save(savename, writer=writer)
        # Fast rendering
        canvas_width, canvas_height = fig.canvas.get_width_height()

        # Open an ffmpeg process
        outf = savename
        cmdstring = ('ffmpeg',
                     '-y', '-r', "%i" % fps,    # overwrite, fps
                     '-s', '%dx%d' % (canvas_width, canvas_height),  \
                                                    # size of image string
                     '-pix_fmt', 'argb',            # format
                     '-f', 'rawvideo',  '-i', '-',  # tell ffmpeg to expect raw
                                                    # video from the pipe
                     '-b:v', '5M',                  # bitrate
                     '-vcodec', 'libx264', outf)    # output encoding
        p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

        # Draw frames and write to the pipe
        for index in range(niter):
            update_lines(index, data, lines, lag, nobjs)
            fig.canvas.draw()

            # extract the image as an ARGB string
            string = fig.canvas.tostring_argb()

            # write to pipe
            p.stdin.write(string)

        # Finish up
        p.communicate()
