
# ---LICENSE-BEGIN - DO NOT CHANGE OR MOVE THIS HEADER
# This file is part of the Neurorobotics Platform software
# Copyright (C) 2014,2015,2016,2017 Human Brain Project
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# ---LICENSE-END

import argparse as ap
import cv2
import numpy as np
import pickle
import time
from typing import Dict

def parse_args():
    """
    Defines the valid commandline options and the variables they are linked to.

    Returns:
        An object which contains the variables which correspond to the
        commandline options.
    """
    dflt_move=4
    parser = ap.ArgumentParser(description='SNN feature detector')
    parser.add_argument('--c1-output', type=str, default='C1_reconstructions/',
                        help='The toplevel output directory for C1\
                        reconstructions')
    parser.add_argument('--delta', metavar='vert', default=dflt_move, type=int,
                        help='The horizontal and vertical distance between the\
                        basic recognizers')
    parser.add_argument('--feature-dir', type=str,
                        help='A directory where the features are stored as images')
    parser.add_argument('--filter', choices=['canny', 'sobel', 'none'],
                        default='none', help='Sets the edge filter to be used.\
                        Defaults to \'none\'')
    parser.add_argument('--frames', default=10, type=int,
                        help='The number of video frames to be processed')
    parser.add_argument('--no-c1', action='store_true',
                        help='Disables the creation of C1 layers')
    parser.add_argument('--plot-spikes', action='store_true',
                        help='Plot the spike trains of all layers')
    parser.add_argument('--plot-weights', action='store_true',
                        help='Plots the learned feature weights and exits')
    parser.add_argument('--refrac-s1', type=float, default=.1, metavar='MS',
                        help='The refractory period of neurons in the S1 layer in ms')
    parser.add_argument('--refrac-s2', type=float, default=.1, metavar='MS',
                        help='The refractory period of neurons in the S2 layer in ms')
    parser.add_argument('--refrac-c1', type=float, default=.1, metavar='MS',
                        help='The refractory period of neurons in the C1 layer in ms')
    parser.add_argument('--reconstruct-s1-img', action='store_true',
                        help='If set, draws a reconstruction of the recognized\
                        features from S1')
    parser.add_argument('--reconstruct-c1-img', action='store_true',
                        help='If set, draws a reconstruction of the recognized\
                        features from C1')
    parser.add_argument('--scales', default=[1.0, 0.71, 0.5, 0.35, 0.25],
                        nargs='+', type=float,
                        help='A list of image scales for which to create\
                        layers. Defaults to [1, 0.71, 0.5, 0.35, 0.25]')
    parser.add_argument('--sim-time', default=100, type=float, help='Simulation time')
    parser.add_argument('--target-name', type=str,
                        help='The name of the already edge-filtered image to\
                              be recognized')
    args = parser.parse_args()
    print(args)
    return args

def filter_img(target_img, filter_type):
    """
    Performs the given edge detector on the given image

    Arguments:
        `target_img`: The image to detect edges from

        `filter_type`: The filter to be applied to the target image. Can be one
                       of 'canny', 'sobel' or 'none', if the image is to be
                       used as-is.

    Returns:
        An image containing the edges of the target image 
    """
    blurred_img = cv2.GaussianBlur(target_img, (5, 5), 1.4)
    filtered_img = None
    if filter_type == 'none':
        return target_img
    if filter_type == 'canny':
        filtered_img = cv2.Canny(blurred_img, 70, 210)
    else:
        dx = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0)
        dy = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1)
        edge_detected = cv2.sqrt(dx * dx + dy * dy)
        filtered_img = cv2.convertScaleAbs(edge_detected)
    return filtered_img

def get_gabor_feature_names():
    """
    Returns the feature names of the gabor filtered images
    """
    return ['slash', 'horiz_slash', 'horiz_backslash', 'backslash']
    
def get_gabor_edges(target_img) -> Dict[str, np.array]:
    """
    Computes the gabor filtered images for four orientations for the given
    unfiltered image

    Parameters:
        `target_img`: The original target image

    Returns:
        A dictionary which contains for each name the corresponding filtered
        image
    """
    angles = [np.pi / 8, np.pi / 4 + np.pi / 8, np.pi / 2 +  np.pi / 8,
              3 * np.pi / 4 + np.pi / 8]
    feature_names = get_gabor_feature_names()
    blurred_img = cv2.GaussianBlur(target_img, (5, 5), 5)
    return dict([(name,
                  cv2.convertScaleAbs(\
                    cv2.filter2D(blurred_img, cv2.CV_64F,
                                cv2.getGaborKernel((5, 5), 1.4, angle, 5, 1))))\
                  for name, angle in zip(feature_names, angles)])

def read_and_prepare_img(target_name, filter_type):
    """
    Reads the input image and performs the edge detector of the passed
    commandline arguments on it

    Arguments:
        `target_name`: The name of the image to be read

        `filter_type`: The filter to be applied to the target image. Can be one
                       of 'canny', 'sobel' or 'none', if the image is to be
                       used as-is.

    Returns:
        An image containing the edges of the target image 
    """
    target_img = cv2.imread(target_name, cv2.CV_8U)
    # Optionally resize the image to 300 pixels (or less) in height
    return filter_img(target_img, filter_type)

def float_to_fourcc_string(x):
    """
    Converns a float to its fourcc number as a string.

    Parameters:
        `x`: The float as returned by cv2.VideoCapture.get(cv2.CAP_PROP_FOURCC)

    Returns:
        The used encoder extension as a string
    """
    x = int(x)
    c1 = chr(x & 0xFF)
    c2 = chr((x & 0xFF00) >> 8)
    c3 = chr((x & 0xFF0000) >> 16)
    c4 = chr((x & 0xFF000000) >> 24)
    return c1 + c2 + c3 + c4

def fourcc_string_to_int(s):
    """
    Converns a fourcc string to a float 

    Parameters:
        `s`: The fourcc string to be converted

    Returns:
        A float representing the code for the given codec string
    """
    n1 = ord(s[0])
    n2 = ord(s[1])
    n3 = ord(s[2])
    n4 = ord(s[3])
    return (n4 << 24) + (n3 << 16) + (n2 << 8) + n1
