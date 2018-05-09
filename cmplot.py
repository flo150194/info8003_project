"""
Developed in collaboration with Gaming1
"""
# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle


def plot_bar(x, y, names=None, x_label=None, y_label=None, grid=False, legend=False):
    """Plot a bar graph of x against y (which may contain more than one function to plot).

    :param x:       (number list) The abscissa points to plot.
    :param y:       ((number list) list) A list of function values to plot. Each sublist must have the same length as x.
    :param names:   (string list: optional, default: None) A list of names which correspond to the functions given in y.
                    names[i] is the name of function y[i].
    :param x_label: (string: optional, default: None) The text to display on the graph's x-axis.
    :param y_label: (string: optional, default: None) The text to display on the graph's y-axis.
    :param grid:    (boolean: optional, default: False) Set to True to display a grid on the graph.
    :param legend:  (boolean: optional, default: False) Set to True to display a legend on the graph. Only relevant when
                    the variable 'names' is not None.
    :return:        Plot a graph using the matplotlib library of the signals contained in y against x.
    """
    bar_width = 0.9 / len(y)
    index = np.arange(len(x))

    for i in range(len(y)):
        if names is not None:
            plt.bar(index + bar_width * (i - np.floor(len(y)/2)), y[i], bar_width, label=names[i])
        else:
            plt.bar(index + bar_width * (i - np.floor(len(y)/2)), y[i], bar_width)
    plt.xticks(index, x)

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if names is not None and legend:
        plt.legend()

    plt.grid(grid)
    plt.tight_layout()
    plt.show()


def plot_graph(x, y, colors, names=None, x_label=None, y_label=None, grid=False, legend=False, y_range=None, show=False,
               new=True):
    """Plot a linear graph of x against y (which may contain more than one function to plot).

    :param x:       (number list) The abscissa points to plot.
    :param y:       ((number list) list) A list of function values to plot. Each sublist must have the same length as x.
    :param names:   (string list: optional, default: None) A list of names which correspond to the functions given in y.
                    names[i] is the name of function y[i].
    :param x_label: (string: optional, default: None) The text to display on the graph's x-axis.
    :param y_label: (string: optional, default: None) The text to display on the graph's y-axis.
    :param grid:    (boolean: optional, default: False) Set to True to display a grid on the graph.
    :param legend:  (boolean: optional, default: False) Set to True to display a legend on the graph. Only relevant when
                    the variable 'names' is not None.
    :param y_range: (number list) Y-axis limits.
    :param show:    (boolean: optional) Set to true to immediately display the graph
    :param new:     (boolean: optiional) Set to true to use a new figure
    :param colors:  (str list: optional) List of colors of the plots
    :return:        Plot a graph using the matplotlib library of the signals contained in y against x.
    """
    if new:
        plt.figure()
    if y_range is not None:
        plt.ylim(y_range)

    for i in range(len(y)):
        if names is not None:
            plt.plot(x, y[i], label=names[i], color=colors[i])
        else:
            plt.plot(x, y[i], color=colors[i])

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if names is not None and legend:
        plt.legend()

    plt.grid(grid)
    plt.tight_layout()

    if show:
        plt.show()


def dot_plot(x, y, colors, names=None, x_label=None, y_label=None, grid=False, y_range=None, legend=False, show=False,
             new=True):
    """Plot a dot graph of x against y (which may contain more than one function to plot).

    :param x:       (number list) The abscissa points to plot.
    :param y:       (number list) Function values to plot. Must have the same length as x.
    :param colors:  (str list)
    :param names:   (string list: optional, default: None) A list of names which correspond to the functions given in y.
                    names[i] is the name of function y[i].
    :param x_label: (string: optional, default: None) The text to display on the graph's x-axis.
    :param y_label: (string: optional, default: None) The text to display on the graph's y-axis.
    :param grid:    (boolean: optional, default: False) Set to True to display a grid on the graph.
    :param legend:  (boolean: optional, default: False) Set to True to display a legend on the graph. Only relevant when
                    the variable 'names' is not None.
    :param y_range: (number list) Y-axis limits.
    :param show:    (boolean: optional) Set to true to immediately display the graph
    :param new:     (boolean: optiional) Set to true to use a new figure
    :return:        Plot a graph using the matplotlib library of the signals contained in y against x.
    """

    fig, ax = plt.subplots()
    for i in range(len(colors)):
        ax.scatter(x[i], y[i], c=colors[i], label=names[i],
                   alpha=0.3, edgecolors='none')

    ax.legend()
    ax.grid(True)

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if names is not None and legend:
        plt.legend()

    plt.grid(grid)
    plt.tight_layout()

    if show:
        plt.show()
