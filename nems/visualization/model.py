import collections.abc
import math
import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .tools import ax_remove_box, ax_bins_to_seconds


_DEFAULT_PLOT_OPTIONS = {
    'skip_plot_options': False,
    'show_x': False, 'xlabel': None, 'xmax_scale': 1, 'xmin_scale': 1,
    'show_y': True, 'ylabel': None, 'ymax_scale': 1, 'ymin_scale': 1,
    'show_seconds': True,
    'legend': False,
    # Right of axis by default, aligned to top
    'legend_kwargs': {
        'frameon': False, 'bbox_to_anchor': (1, 1), 'loc': 'upper left'
        },
    }

def set_plot_options(ax, layer_options, time_kwargs=None):
    """Adjust matplotlib axes object in-place according to `layer_options`.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    layer_options : dict
        Layer-specific options with the same dictionary structure as
        `_DEFAULT_PLOT_OPTIONS`. Layer-specific options take precedence.
    time_kwargs : dict; optional.
        Keyword arguments for `.tools.ax_bins_to_seconds`.

    Returns
    -------
    ops : dict
        Merged plot options from `_DEFAULT_PLOT_OPTIONS` and `layer_options`.

    See also
    --------
    nems.layers.base.Layer.plot_options
    
    """
    _dict = copy.deepcopy(_DEFAULT_PLOT_OPTIONS)
    ops = _nested_update(_dict, layer_options)
    if ops['skip_plot_options']: return

    # x-axis
    ax.xaxis.set_visible(ops['show_x'])
    ax.set_xlabel(ops['xlabel'])
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin*ops['xmin_scale'], xmax*ops['xmax_scale'])

    # y-axis
    ax.yaxis.set_visible(ops['show_y'])
    ax.set_ylabel(ops['ylabel'])
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin*ops['ymin_scale'], ymax*ops['ymax_scale'])

    # Convert bins -> seconds
    if (time_kwargs is not None) and (ops['show_seconds']):
        ax_bins_to_seconds(ax, **time_kwargs)

    # Add legend
    if ops['legend']:
        ax.legend(**ops['legend_kwargs'])
    
    # Remove top and right segments of border around axes
    ax_remove_box(ax)

    return ops


def _nested_update(d, u):
    """Merge two dictionaries that may themselves contain nested dictionaries.
    
    Internal for `set_plot_options`. Using this so that `Layer.plot_options`
    can update some keys of nested dicts (like 'legend_kwargs') without losing
    defaults for other keys.
    TODO: maybe move this to generic NEMS tools? Could be useful elsewhere.
    
    References
    ----------
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def plot_model(model, input, target=None, n=None, select_layers=None,
               n_columns=1, show_titles=True, figure_kwargs=None,
               sampling_rate=None, time_axis='x', conversion_factor=1,
               decimals=2,  **eval_kwargs):
    """Plot result of `Layer.evaluate` for each Layer in a Model.
    
    Aliased as `Model.plot_output()`.
    TODO: Insert plot of input on first axis, as with old NEMS? But `input` may
          not always be a single spectrogram now, so user would have to specify
          which array to plot, whether to use imshow vs plot vs something else,
          etc... may not be worth it. Alternatively, could add option to insert
          subfigures (possibly at arbitrary positions?) with user-generated
          content, but aligning the x-axis for those could be tricky.

    Parameters
    ----------
    model : Model
        See `nems.models.base.Model`.
    input : ndarray or dict.
        Input data for Model. See `Model.evaluate` for expected format.
    target : ndarray or list of ndarray; optional.
        Target data for Model. See `Model.fit` for expected format. If provided,
        target(s) will be shown on the last axis of the final layer's figure.
    n : int or None; optional.
        Number of Layers to plot. Defaults to all.
    select_layers : str, int, slice, or list; optional.
        Selects layers to plot. Any valid `index` for `Model.layers[index]` is
        acceptable. Ex: `layers = slice(1,3)` to only plot the Layers at indices
        1 and 2, or `layers = ['fir', 'dexp']` to only plot the Layers named
        'fir' and 'dexp'.
        NOTE: This option supercedes `n` if neither is None.
    n_columns : int; default=1.
        Number of columns to arrange plots on (all columns within a row will
        be filled before moving to the next row). Note that values > 1 will
        reduce the width of subplots slightly to make room for the default
        legend placement. However, it may still be necessary to increase
        figure size if the number of columns is large.
    show_titles : bool; default=True.
        Specify whether to show `Layer.name` as a title above each Layer's
        subfigure.
    figure_kwargs : dict or None; optional.
        Keyword arguments for `matplotlib.pyplot.figure`.
        Ex: `figure_kwargs={'figsize': (10,10)}`.
    sampling_rate : float; optional.
        Sampling rate (in Hz) for `input` (and `target` if given), used to
        convert time_bin labels to seconds (if not None).
    time_axis : str; default='x'.
        'x' or 'y', the axis on the resulting Matplotlib axes objects that
        represents time.
    conversion_factor : int; default=1.
        Multiply seconds by this number to get different units. Floating point
        values must correspond to rational numbers. Scientific notation may only
        be used for multiples of 10, and will be converted to an integer.
        Ex: `conversion_factor=1000` to get units of milliseconds.
            `conversion_factor=1/60` to get units of minutes.
    decimals : int; default=2.
        Number of decimal places to show on new tick labels.
    eval_kwargs : dict; optional.
        Additional keyword arguments to supply to `Model.evaluate`.
        Ex: `input_name='stimulus'`.

    Returns
    -------
    fig, axes : (matplotlib.figure.Figure, matplotlib.axes.Axes)
    
    See also
    --------
    nems.models.base.Model
    nems.layers.base.Layer.plot
    nems.visualization.tools.ax_bins_to_seconds

    """

    # Collect options for `ax_bins_to_seconds`
    if sampling_rate is not None:
        time_kwargs = {
            'sampling_rate': sampling_rate, 'time_axis': time_axis,
            'conversion_factor': conversion_factor, 'decimals': decimals
            }
    else:
        time_kwargs = None

    # Determine which figures to plot and set figure layout.
    # One subfigure per Layer, all in a single column by default
    # (otherwise fill columns before rows).
    figure_kwargs = {} if figure_kwargs is None else figure_kwargs
    figure = plt.figure(**figure_kwargs)
    if select_layers is not None:
        layers = model.layers.__getitem__(select_layers)
        if not isinstance(layers, list):
            layers = [layers]
    else:
        layers = model.layers[:n]
    n_rows = math.ceil(len(layers)/n_columns)
    subfigs = figure.subfigures(n_rows, n_columns)
    if isinstance(subfigs, matplotlib.figure.SubFigure):
        # Only one subfigure
        subfigs = np.array([subfigs])

    # Evaluate the model with `save_layer_outputs=True` to make sure all
    # intermediate Layer outputs are stored.
    data = model.evaluate(input, save_layer_outputs=True, **eval_kwargs)
    layer_outputs = data['_layer_outputs']

    # Re-order layer outputs so that each index of `outputs` is a list
    # containing all of the outputs from one layer, while keeping the output
    # lists in Layer-order.
    outputs = []
    for i, layer in enumerate(layers):
        outputs.append([])
        for k, v in layer_outputs.items():
            if k.startswith(layer.name):
                outputs[i].append(v)

    # Generate the plot for each layer
    iter_zip = zip(outputs, layers, subfigs.flatten())
    for i, (output, layer, subfig) in enumerate(iter_zip):
        layer.plot(output, fig=subfig, **layer.plot_kwargs)
        if show_titles:
            # Make room for titles
            subfig.subplots_adjust(top=0.8)
            name = layer.name
            subfig.suptitle(f'({model.get_layer_index(name)}) {name}')
        if n_columns > 1:
            # For multiple columns, shrink width of axes a bit to make room for
            # default legend placement.
            subfig.subplots_adjust(right=0.8)
        for ax in subfig.axes:
            set_plot_options(ax, layer.plot_options, time_kwargs=time_kwargs)

    # Add alternating white/gray shading to make Layers easier to distinguish.
    for shaded_fig in subfigs[checkerboard(subfigs)]:
        shaded_fig.patch.set_facecolor((0, 0, 0, 0.075))  # black, low alpha

    # Final x-axis of the final layer in each column is always visible
    # so that time is visually synchronized for all plots above.
    if len(subfigs.shape) == 1:
        bottom_figs = [subfigs[-1]]
    else:
        bottom_figs = subfigs[-1,:]
    for bottom_fig in bottom_figs:
        for ax in bottom_fig.axes:
            ax.xaxis.set_visible(True)

    # Add plot of target if given, on last axis of last subfig.
    last_ax = subfig.axes[-1]
    if target is not None:
        if not isinstance(target, list):
            target = [target]
        for i, y in enumerate(target):
            last_ax.plot(y, label=f'Target {i}')
        last_ax.legend(**_DEFAULT_PLOT_OPTIONS['legend_kwargs'])
        last_ax.autoscale()

    return figure


def plot_layer(output, fig=None, **plot_kwargs):
    """Default Layer plot, displays all outputs on a single 2D line plot.
    
    Parameters
    ----------
    output : ndarray or list of ndarray
        Return value of `Layer.evaluate`.
    fig : matplotlib.pyplot.figure.Figure; optional.
        Matplotlib Figure to render the plot on. If not provided, a new figure
        will be generated with default options.
    plot_kwargs : dict
        Additional keyword arguments for `matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.figure.Figure

    See also
    --------
    .plot_model

    """
    if not isinstance(output, list):
        output = [output]
    if fig is None:
        fig = plt.figure()
    ax = fig.subplots(1,1)
    plot_data = np.concatenate(output, axis=1)
    ax.plot(plot_data, **plot_kwargs)

    return fig


def checkerboard(array):
    """Get checkerboard-spaced indices for a numpy array.

    Using this to keep background-shading of subfigures alternately-spaced.

    Parameters
    ----------
    array : ndarray
    
    Returns
    -------
    indices : boolean ndarray
        Same shape as array, with True and False in a checkerboard pattern.

    References
    ----------
    From user Eelco Hoogendoorn
    https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy

    Examples
    --------
    >>> x = np.ones(shape=(3,3))
    >>> checkerboard(x)
    array([[False,  True, False],
           [ True, False,  True],
           [False,  True, False]])
    >>> x[checkerboard(x)] = 2
    array([[1., 2., 1.],
           [2., 1., 2.],
           [1., 2., 1.]])

    """
    shape = array.shape
    indices = (np.indices(shape).sum(axis=0) % 2).astype(bool)
    return indices