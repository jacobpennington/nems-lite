import collections.abc
import math
import copy

import numpy as np
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
    
    TODO: document parameters

    Returns
    -------
    ops : dict
        Merged plot options from `_DEFAULT_PLOT_OPTIONS` and `layer_options`.
    
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


def plot_model(model, input, target=None, n=None, n_columns=1,
               sampling_rate=None, time_axis='x', conversion_factor=1,
               decimals=2, show_titles=True, figure_kwargs=None, **eval_kwargs):
    """Plot result of `Layer.evaluate` for each Layer in a Model.
    
    Aliased as `Model.plot_output()`.

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
    n_columns : int; default=1.
        Number of columns to arrange plots on (all columns within a row will
        be filled before moving to the next row).
    sampling_rate : float; optional.
        Sampling rate (in Hz) for `input` (and `target` if given), used to
        convert time_bin labels to seconds (if not None).
    time_axis : str; default='x'.
        'x' or 'y', the axis on the resulting Matplotlib axes objects that
        represents time.
    conversion_factor : float; default=1.
        Multiply seconds by this number to get different units. E.g.
        `conversion_factor=1e3` to get milliseconds.
    decimals : int
        Number of decimal places to show on new tick labels
    show_titles : bool; default=True.
        Specify whether to show `Layer.name` as a title above each Layer's
        subfigure.
    figure_kwargs : dict or None; optional.
        Keyword arguments for `matplotlib.pyplot.figure`.
        Ex: `figure_kwargs={'figsize': (10,10)}`.
    eval_kwargs : dict; optional.
        Additional keyword arguments to supply to `Model.evaluate`.
        Ex: `input_name='stimulus'`.

    Returns
    -------
    fig, axes : (plt.figure.Figure, matplotlib.axes.Axes)
    
    See also
    --------
    nems.models.base.Model
    nems.layers.base.Layer.plot

    """
    figure_kwargs = {} if figure_kwargs is None else figure_kwargs
    if sampling_rate is not None:
        time_kwargs = {
            'sampling_rate': sampling_rate, 'time_axis': time_axis,
            'conversion_factor': conversion_factor, 'decimals': decimals
            }
    else:
        time_kwargs = None

    layers = model.layers[:n]
    n_rows = math.ceil(len(layers)/n_columns)
    figure = plt.figure(**figure_kwargs)
    subfigs = figure.subfigures(n_rows, n_columns)

    # Evaluate the model with `save_layer_outputs=True` to make sure all
    # intermediate Layer outputs are stored.
    data = model.evaluate(input, save_layer_outputs=True, **eval_kwargs)
    layer_outputs = data['_layer_outputs']

    # Re-order layer outputs so that each index of `outputs` is a list
    # containing all of the outputs from one layer, while keeping the output
    # lists in Layer-order.
    outputs = []
    i = -1
    current_name = ''
    for k, v in layer_outputs.items():
        # k should have {layer.name}.{data_key}.{i} format
        if current_name.startswith(k.split('.')[0]):
            # Same Layer, so append data to the current output
            outputs[i].append(v)
        else:
            # Start a new output list for next Layer
            i += 1
            outputs.append([v])

    shading = False
    for output, layer, subfig in zip(outputs, layers, subfigs.flatten()):
        layer.plot(output, fig=subfig, **layer.plot_kwargs)
        if show_titles:
            subfig.subplots_adjust(top=0.8)  # make some room
            subfig.suptitle(layer.name)
        for ax in subfig.axes:
            set_plot_options(ax, layer.plot_options, time_kwargs=time_kwargs)
        # Alternate white/gray background to see the layer breaks easier
        # (otherwise could get confusing with some layers using multiple plots).
        if shading:
            subfig.patch.set_facecolor((0,0,0,0.075))
        shading = not shading

    # First x-axis of final layer is always visible
    last_ax = subfig.axes[-1]
    last_ax.xaxis.set_visible(True)
    # Add plot of target if given, on last axis of last subfig
    if target is not None:
        if not isinstance(target, list):
            target = [target]
        for i, y in enumerate(target):
            last_ax.plot(y, label=f'Target {i}')
        last_ax.legend(**_DEFAULT_PLOT_OPTIONS['legend_kwargs'])
        last_ax.autoscale()

    return figure


def plot_layer(layer, output, fig=None, **plot_kwargs):
    """Default Layer plot.
    
    TODO: docs
    
    """
    if fig is None:
        fig = plt.figure()
    ax = fig.subplots(1,1)
    plot_data = np.concatenate(output, axis=1)
    ax.plot(plot_data, **plot_kwargs)

    return fig
