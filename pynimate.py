"""
Created on Mon Jun 04 21:11:10 2018

@autor: pierm

Module that tries making data animation easy
Support for anims of several subplots coming soon..
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
import matplotlib.animation as animation
import sys

__version__ = "2.1"


def anim(data, order="iod", colors=None, xlim=None, ylim=None, zlim=None,
         sigma=3, labels=None, label_kwargs=None, interval=40, lag=1,
         fade=False, plotdims=None, flow=True, scatter=True,
         scatter_kwargs=None, traces=True, traces_kwargs=None, fig=None,
         ax=None, keeplims=False, save=False, savename='animation.mp4'):
    """
    Animates 1d/2d/3d data using matplotlib.animation

    Parameters
    ----------
    data : array_like(rank={1,2,3})
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
        If a wrong character is provided, a KeyError is raised

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
        Array with [ymin, ymax]. Check xlim. If it is provided with ndims=1
        then it is ignored

    zlim : array_like[2]/tuple, optional
        Array with [ymin, ymax]. Check xlim. If it is provided with ndims={1,2}
        then it is ignored

    sigma : int/float, optional
        If lims are not provided, 'sigma' represents the number of stds from
        the mean value; this sets the plot limits.
        Examples: sigma=3 - > xlim=(mean-3*std,mean+3*std)

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

    # fade : bool, optional
    #     TODO with LineCollection
    #     Make the data slowly disappear by lowering `alpha` (if lag>1)

    plotdims : array_like(1d/2d)(ints)(len=1,2,3), optional
        The indices of the params you wish to plot (order counts, so the first
        one will be on the x axis , the second on the y axis  etc.).
        If None, default is [0,1,2] etc. but make sure pass a single ax if None
        If len(plotdims)<0 or > 3 a ValueError is raised.
        If len(plotdims)>ndims, plotdims is trimmed -> plotdims[0:ndims]
        If plotdims contains two equal values, a ValueError is raised
        Example: plotdims=[0,2,1] plots param 0 on x, param 2 on y, param 1 on
        z.
        If rank(plotdims)==2 then multiple subplot animation is enabled, so
        len(plotdims) must be equal to len(ax).
        Example:
            plotdims=[[0,1], [2]]
            ax=(ax1, ax2)
            This will animate params 0,1 on ax1 and param 2 on ax2

    flow : bool, optional
        Animation control for ndims=1.
        If True, data is plotted in 2d (where the extra dimension is time).
        If False, animation is in 1d (not cool :( )

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

    ax : matplotlib.axes, array_like/tuple[1d] of matplotlib.axes, optional
        A axes instance on which to add the animation.
        If None, it creates a new subtplots instance. If it is not None but
        fig is, it overrides ax to None.
        If a tuple of axes is provided, animation is performed on all subplots

    keeplims : bool, optional
        Override the previous ax limits if ax is provided

    save : bool, optional
        Set True if you want to save the animation on file.

    savename : string, optional
        Name of the rendered file

    # TODO: Add more arguments to anim (maybe like a dictionary)

    Returns
    -------
    fig : matplotlib.Figure
        Figure where everything is plotted/animated. It might be useful if you
        want to change some properties, but remember that if you do that the
        animation will stop.


    Notes
    -----
    If you are planning to animate a subplot whose figure has more than one
    subplot, call anim after you plotted all the static subplots.
    """

    def set_limits(xlim, ylim, zlim, ax, data, sigma, plotdim):
        """
        Sets the limits of a matplotlib.axes instance. Needed by pynimate.anim
        If limits are not provided, plot [mean +/- sigma*std] for each
        component
        """
        ndims = len(plotdim)
        if xlim is None:
            xmean = np.mean(data[:, :, plotdim[0]])
            xstd = np.std(data[:, :, plotdim[0]])
            xlim = xmean-sigma*xstd, xmean+sigma*xstd

        if ndims == 1 or ndims == 2:
            ax.set_xlim(xlim)
        elif ndims == 3:
            ax.set_xlim3d(xlim)
        else:
            raise ValueError("'xlim' not understood")

        if ylim is None:
            if ndims == 2 or ndims == 3:
                ymean = np.mean(data[:, :, plotdim[1]])
                ystd = np.std(data[:, :, plotdim[1]])
                ylim = ymean-sigma*ystd, ymean+sigma*ystd

        if ndims == 1:
            ax.set_ylim(-0.5, 1.5)
        elif ndims == 2:
            ax.set_ylim(ylim)
        elif ndims == 3:
            ax.set_ylim3d(ylim)
        else:
            raise ValueError("'ylim' not understood")

        if ndims == 3:
            zmean = np.mean(data[:, :, plotdim[2]])
            zstd = np.std(data[:, :, plotdim[2]])
            zlim = zmean-sigma*zstd, zmean+sigma*zstd

            try:
                ax.set_zlim3d(zlim)
            except ValueError:
                print("`zlim` not understood")
                sys.exit(1)

        return ax

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

    if len(order) > 3 or data.ndim != len(order):
        raise ValueError("Size mismatch in 'order' argument")

    # Find the format of the provided data, getting various indices
    indices = dict(i=None, o=None, d=None)
    for i, character in enumerate(order):
        if indices[character] is None:  # KeyError raised if there is a problem
            indices[character] = i
        else:
            raise ValueError("'order' argument has repetitions!")

    # Adding the missing index as the first dimension of data
    for key in indices:
        if indices[key] is None:
            data = np.array([data])
            indices[key] = 0
            for key2 in indices:
                if key2 != key and indices[key2] is not None:
                    indices[key2] += 1

    iterindex = indices['i']
    objindex = indices['o']
    dimindex = indices['d']

    niter = data.shape[iterindex]
    nobjs = data.shape[objindex]
    ndims = data.shape[dimindex]

    if plotdims is None:
        # If plotdims is None then choose axis in the usual way...
        if ndims <= 3:
            plotdims = [[i for i in range(ndims)]]
        # unless if the user provided too many dimensions!
        else:
            raise ValueError("Plot for ndims > 3 is allowed, but you need to "
                             "provide the params you want to plot through "
                             "`plotdims` argument!")
    else:
        plotdimstemp = np.atleast_2d(plotdims)
        if plotdimstemp.dtype == 'O':  # i.e. if plotdims=[[0,1],[1,3,2]]
            for i in range(len(plotdims)):
                plotdims[i] = np.array(plotdims[i])
        else:
            plotdims = plotdimstemp
        # plotdims = [np.array(plotdims[i]) for i in range(len(plotdims))]
        for plotdim in plotdims:
            if np.any(plotdim > (ndims - 1)) or np.any(plotdim < 0):
                raise ValueError("'plotdims' is asking for a dimension that "
                                 "does not exist!")
        # Avoiding problems with lines of different lenght
            if len(plotdim) != len(np.unique(plotdim)):
                raise ValueError("Value repetition in plotdims is not allowed")

    if labels is not None:
        if len(labels) < ndims:
            raise ValueError("You did't provide enough labels for the axes! "
                             "n_labels=%i, while ndims = %i"
                             % (len(labels), ndims))

    # Overriding ndims to match dims required. This is ok since plotdims
    # was previously "allocated" if it was None.
    # TODO : fix ndims for more axes
    ndims = len(plotdims[0])  # Not worrying about different dimensions...
    if ndims < 1 or ndims > 3:
        # but you still need to plot 1d, 2d or 3d animations!
        raise ValueError("You can't animate less than 1d or more than 3d!")

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

    def update_lines(index, data, lines, lag, nobjs, plotdims):
        # This is needed to allow the user to make multiple animations on
        # different subplots
        for k, plotdim in enumerate(plotdims):
            nlines = len(lines)/len(plotdims)
            ndims_plot = len(plotdim)
            for j in range(nlines):
                firstindex = np.maximum(0, index - lag)  # Lag

                if ndims_plot == 1:
                    ydata = np.zeros(np.minimum(index, lag))
                    if flow:  # Flowing plot
                        ydata = np.arange(len(ydata), dtype=float)[::-1]\
                                / (lag - 1)
                    lines[k*nlines+j].set_data(
                            data[firstindex:index, j % nobjs, plotdim[0]],
                            ydata)
                elif ndims_plot == 2:
                    # j%nobjs is to plot traces as well
                    lines[k*nlines+j].set_data(
                            data[firstindex:index, j % nobjs, plotdim[0]],
                            data[firstindex:index, j % nobjs, plotdim[1]])
                elif ndims_plot == 3:
                    # Traspose because the shape required is dim*points for
                    # each dim
                    # i.e. 3*10 if lag is 10 and ndims is 3
                    # 0:2 sets xaxis and yaxis
                    lines[k*nlines+j].set_data(
                            data[firstindex:index, j % nobjs, plotdim[0:2]].T)
                    lines[k*nlines+j].set_3d_properties(
                            data[firstindex:index, j % nobjs, plotdim[2]].T)
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

    # Create a new figure if one wasn't provided
    if fig is None:
        if keeplims:
            raise ValueError("You can't use 'keeplims' if you don't provide "
                             "your own ax!")
        got_figure = False
        if ndims == 1 or ndims == 2:
            fig, ax = plt.subplots()
        elif ndims == 3:
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        else:
            # Error Handling was done already, but you never know...
            raise SystemExit(0)
    else:
        got_figure = True
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
    axlist = np.atleast_1d(ax)

    # NOTE : Make sure that if you extend this you have to modify update_line
    if len(plotdims) != len(axlist):
        raise ValueError("'plotdims' must be the same lenght of ax!")

    # Set lims if no fig was provided or if the user wants to keep the old lims
    # of the ax he provided
    if not keeplims or not got_figure:
        for j, ax in enumerate(axlist):
            ax = set_limits(xlim, ylim, zlim, ax, data, sigma, plotdims[j])

    # Setting labels
    if labels is not None:
        # for j, ax in enumerate(axlist):
        for ax in axlist:
            # TODO : multi labels
            ax.set_xlabel(labels[plotdims[0][0]], **label_kwargs)
            if ndims == 2 or ndims == 3:
                ax.set_ylabel(labels[plotdims[0][1]], **label_kwargs)
            if ndims == 3:
                ax.set_zlabel(labels[plotdims[0][2]], **label_kwargs)

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

    lines = []
    for j, ax in enumerate(axlist):
        # Create scatter `lines` only if the user wants to plot points
        if scatter:
            if ndims == 1 or ndims == 2:
                for i in range(nobjs):
                    lines.append(ax.plot([], [], 'o', color=colors[i],
                                 **scatter_kwargs)[0])
            elif ndims == 3:
                for i in range(nobjs):
                    lines.append(ax.plot(data[0:1, i, plotdims[j][0]],
                                         data[0:1, i, plotdims[j][1]],
                                         data[0:1, i, plotdims[j][2]], 'o',
                                         color=colors[i], **scatter_kwargs)[0])

            # Saving automatic colors to pass to traces
            for i in range(nobjs):
                colors[i] = lines[i].get_color()

        # Add lines for traces
        if traces:
            for i in range(nobjs):
                if ndims == 1 or ndims == 2:
                    lines.append(ax.plot([], [], color=colors[i],
                                 **traces_kwargs)[0])
                elif ndims == 3:
                    lines.append(ax.plot(data[0:1, i, plotdims[j][0]],
                                         data[0:1, i, plotdims[j][1]],
                                         data[0:1, i, plotdims[j][2]],
                                         color=colors[i], **traces_kwargs)[0])

    def show_anim():
        # Just a quick fix using global variables to avoid nasty loops later
        global ln_an
        if ndims == 1 or ndims == 2:
            ln_an = animation.FuncAnimation(fig, update_lines, init_func=init,
                                            fargs=(data, lines, lag, nobjs,
                                                   plotdims),
                                            interval=interval, frames=niter+1,
                                            blit=True)
        else:
            ln_an = animation.FuncAnimation(fig, update_lines,
                                            fargs=(data, lines, lag, nobjs,
                                                   plotdims),
                                            interval=interval, frames=niter+1,
                                            blit=True)

    if not save:
        # Finally animate
        show_anim()
        plt.show()
    else:
        is_error = False
        # Save animation
        fps = 1000/interval
        try:
            import subprocess   # Try to import this for really fast rendering
        except ImportError:
            print("Import subprocess failed. Rendering will be slower")
            show_anim()
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, bitrate=180,
                            metadata=dict(artist='pynimate by pjoe95'))
            ln_an.save(savename, writer=writer)
            is_error = True
            print("Done")

        if not is_error:
            # Fast rendering
            canvas_width, canvas_height = fig.canvas.get_width_height()

            # Open an ffmpeg process
            outf = savename
            cmdstring = ('ffmpeg',
                         '-y', '-r', "%i" % fps,    # overwrite, fps
                         '-s', '%dx%d' % (canvas_width, canvas_height),  \
                         # size of image string
                         '-pix_fmt', 'argb',            # format
                         '-f', 'rawvideo',  '-i', '-',  # tell to expect raw
                                                        # video from the pipe
                         '-b:v', '5M',                  # bitrate
                         '-vcodec', 'libx264', outf)    # output encoding
            p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

            # Draw frames and write to the pipe
            for index in range(niter):
                update_lines(index, data, lines, lag, nobjs, plotdims)
                fig.canvas.draw()

                # extract the image as an ARGB string
                string = fig.canvas.tostring_argb()

                # write to pipe
                p.stdin.write(string)

            # Finish up
            p.communicate()
    return fig
