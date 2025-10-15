"""Loads dataset that is querable by batch."""

import os
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def setup_greener_color():
    cdict1 = {'red':   ((0.0, 0.0, 0.0),
                       (0.00001, 0.1, 0.1),
                       (0.1,0.3,0.3),
                       (1.0, 1.0, 1.0)),

             'green': ((0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

             'blue':  ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
            }
    
#     cdict1 = {'red':   ((0.0, 0.0, 0.0),
#                        (1.0, 0.0, 0.0)),

#              'green': ((0.0, 0.0, 0.0),
#                        (0.00001, 0.1, 0.1),
#                        (0.1,0.3,0.3),
#                        (1.0, 1.0, 1.0)),

#              'blue':  ((0.0, 0.0, 0.0),
#                        (1.0, 0.0, 0.0))
#             }
    

    greener = LinearSegmentedColormap('greener', cdict1)

    x = np.arange(0, np.pi, 0.1)
    y = np.arange(0, 2*np.pi, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(X) * np.sin(Y) * 10
    return greener

# def visualize(data, led_list):
#     cmap = setup_greener_color()
#     Nmeas = led_list.shape[1]
# #     fig = plt.figure(figsize=(8*Nmeas//6, 5))
#     fig = plt.figure(figsize=(25, 5))
#     # TODO (kellman): check this is visually correct
#     vmin = 0
#     vmax = np.max(data)


#     for dd in range(Nmeas):
#         plt.subplot(int(np.ceil(Nmeas/6)), 6, dd+1)
#         patt = data/np.max(data)
#         circle = plt.Circle((0, 0), 0.13, alpha=0.3,facecolor=[1,1,1],edgecolor=None,zorder=10)
#     #         circle2 = plt.Circle((0, 0), metadata['na_illum']*1, alpha=0.3,facecolor=None,edgecolor=None,zorder=9)
#         circle2 = plt.Circle((0, 0), 0.4, alpha=0.2,facecolor=[1,1,1],edgecolor=None,zorder=9)
#         circle3 = plt.Circle((0, 0), 0.45, alpha=1, color='k',edgecolor=None)

#         plt.gca().add_patch(circle3)
#         plt.scatter(led_list[0,:],led_list[1,:],c=patt,marker='o',cmap=cmap,vmin=vmin,vmax=vmax,zorder=8)
#         plt.gca().add_patch(circle)
#         plt.gca().add_patch(circle2)
#         plt.axis('off')
#         plt.axis('equal')
        
#     return fig



def visualize(data, led_list):
    cmap = setup_greener_color()
    
    Nmeas = data.shape[0]
#     fig = plt.figure(figsize=(8*Nmeas//6, 5))
    fig = plt.figure(figsize=(20, 20))
    # TODO (kellman): check this is visually correct
    vmin = 0
    vmax = 1 #np.max(data)

    for dd in range(Nmeas):
        plt.subplot(int(np.ceil(Nmeas/6)), 6, dd+1)
        # patt = data[dd,:]/np.max(data[dd,:])
        patt = data[dd,:]
        circle = plt.Circle((0, 0), 0.13, alpha=0.3,facecolor=[1,1,1],edgecolor=None,zorder=4)
        circle2 = plt.Circle((0, 0), 0.4, alpha=0.2,facecolor=[1,1,1],edgecolor=None,zorder=3)
        circle3 = plt.Circle((0, 0), 0.45, alpha=1, color='k',edgecolor=None)
        plt.gca().add_patch(circle3)
        plt.scatter(led_list[0,:],led_list[1,:],c=patt,marker='o',cmap=cmap,vmin=vmin,vmax=vmax,zorder=4)
        plt.gca().add_patch(circle)
        plt.gca().add_patch(circle2)
        plt.axis('off')
        plt.axis('equal')
    return fig


def visualize_square(data, led_list):
    cmap = setup_greener_color()
    Nmeas = data.shape[0]
#     fig = plt.figure(figsize=(8*Nmeas//6, 5))
    fig = plt.figure(figsize=(20, 20))
    # TODO (kellman): check this is visually correct
    vmin = 0
    vmax = 1 #np.max(data)

    for dd in range(Nmeas):
        plt.subplot(int(np.ceil(Nmeas/6)), 6, dd+1)
        # patt = data[dd,:]/np.max(data[dd,:])
        patt = data[dd,:]
        # circle = plt.Rectangle((-0.13, -0.13),0.26, 0.26, alpha=0.3,facecolor=[1,1,1],edgecolor=None,zorder=4)
        circle = plt.Circle((0, 0), 0.13, alpha=0.3,facecolor=[1,1,1],edgecolor=None,zorder=4)
        circle2 = plt.Rectangle((-0.4, -0.4),0.8, 0.8, alpha=0.2,facecolor=[1,1,1],edgecolor=None,zorder=3)
        circle3 = plt.Rectangle((-0.45, -0.45),0.9, 0.9, alpha=1, color='k',edgecolor=None)
        
        # circle = plt.Rectangle((0, 0), 0.13, alpha=0.3,facecolor=[1,1,1],edgecolor=None,zorder=4)
        # circle2 = plt.Rectangle((0, 0), 0.4, alpha=0.2,facecolor=[1,1,1],edgecolor=None,zorder=3)
        # circle3 = plt.Rectangle((0, 0), 0.45, alpha=1, color='k',edgecolor=None)
        plt.gca().add_patch(circle3)
        plt.scatter(led_list[0,:],led_list[1,:],c=patt,marker='o',cmap=cmap,vmin=vmin,vmax=vmax,zorder=4)
        plt.gca().add_patch(circle)
        plt.gca().add_patch(circle2)
        plt.axis('off')
        plt.axis('equal')
    return fig