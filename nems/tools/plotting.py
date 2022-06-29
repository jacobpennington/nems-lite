import matplotlib.pyplot as plt


def ax_remove_box(axes=None):
    """Remove right and top lines from plot border on Matplotlib axes."""
    if axes is None:
        axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
