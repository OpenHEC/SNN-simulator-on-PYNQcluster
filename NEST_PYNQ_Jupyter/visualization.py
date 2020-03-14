
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

from typing import Dict, Sequence, List
import numpy as np
import pathlib as plb
import pyNN.utility.plotting as plt
import matplotlib.pyplot as mplt
import network as nw
import cv2

def copy_to_visualization(pos, ratio, feature_img, visualization_img,
                          layer_shape, delta, overfull=True):
    """
    Copies a feature image onto the visualization canvas at the given position
    of the neuron layer with the given intensity.

    Arguments:
        `pos`: The position where the feature should be painted

        `ratio`: The intensity ratio with which the feature should be painted

        `feature_img`: The feature which should be drawn

        `visualization_img`: The image onto which to paint the feature

        `layer_shape`: The shape of the neuron layer which detects the features

        `delta`: The horizontal and vertical offset between the feature layers
    """
    n, m = layer_shape
    f_n = feature_img.shape[0]
    f_m = feature_img.shape[1]
    t_n = visualization_img.shape[0]
    t_m = visualization_img.shape[1]
    p_i = int(pos / m)
    p_j = pos % m
    start_i = delta * p_i
    start_j = delta * p_j
    if overfull and p_i == n - 1:
        start_i = t_n - f_n
    if overfull and p_j == m - 1:
        start_j = t_m - f_m
    rationated_feature_img = ratio * feature_img
    for i in range(f_n):
        for j in range(f_m):
            visualization_img[start_i + i][start_j + j] +=\
                rationated_feature_img[i][j]

def plot_weights(weights_dict):
    weight_panels = []
    for name, (weights, shape) in weights_dict.items():
        weight_panels.append(pynnplt.Panel(weights.reshape(10, -1), cmap='gray',
                                       xlabel='{} Connection weights'.format(name),
                                       xticks=True, yticks=True))

    pynnplt.Figure(*weight_panels).save('plots/weights_plot_blurred.png')

def visualization_parts(target_img_shape, layers_dict, feature_imgs_dict,
                        delta, canvas=None):
    """
    Reconstructs the initial image by drawing the features on the recognized
    positions with an intensity proportional to the respective neuron firing rate.

    Parameters:
        `target_image_shape`: The shape of the original image

        `layers_dict`: A dictionary containing for each image scale all feature
                       layers after recording, one layer for each feature.

        `feature_imgs_dict`: A dictionary containing for each name the
                             corresponding feature image

        `delta`: The horizontal and vertical offset between the feature layers

    Returns:
        A dictionary which contains for each size a list of pairs of the
        reconstructed images and their feature name, one pair for each feature
    """
    t_n, t_m = target_img_shape
    three_channels = True
    if canvas == None:
        three_channels = False
        visualization_img = np.zeros( (t_n, t_m) )
    else:
        visualization_img = canvas
    max_firing = 1
    for size, layers in layers_dict.items():
        for layer in layers:
            spike_counts = layer.current_spike_counts
            for count in spike_counts:
                if count > max_firing:
                    max_firing = count
    partial_reconstructions_dict = {}   # size -> list of pairs of reconstructed
                                        # images and their feature name
    for size, layers in layers_dict.items():
        partial_reconstructions_dict[size] = []
        if three_channels:
            scaled_vis_img = np.zeros( (round(t_n * size), round(t_m * size), 3) )
        else:
            scaled_vis_img = np.zeros( (round(t_n * size), round(t_m * size)) )
        for layer in layers:
            print('scale: {}, layer: {}'.format(size, layer.population.label))
            spike_counts = layer.current_spike_counts
            feature_label = layer.population.label
            feature_img = feature_imgs_dict[feature_label]
            st_n = scaled_vis_img.shape[0]
            st_m = scaled_vis_img.shape[1]
            for i in range(layer.population.size):
                # each spiketrain corresponds to a layer S1 output neuron
                copy_to_visualization(i, spike_counts[i] / max_firing,
                                      feature_img, scaled_vis_img, layer.shape,
                                      delta)
            upscaled_vis_img = cv2.resize(src=scaled_vis_img, dsize=(t_m, t_n),
                                          interpolation=cv2.INTER_CUBIC)
            partial_reconstructions_dict[size].append(\
                (visualization_img + upscaled_vis_img.astype(np.int32),
                 feature_label))
            if three_channels:
                scaled_vis_img = np.zeros( (round(t_n * size), round(t_m * size), 3) )
            else:
                scaled_vis_img = np.zeros( (round(t_n * size), round(t_m * size)) )
    return partial_reconstructions_dict

def create_S1_feature_image(target_img, layer_collection, feature_imgs_dict,
                            args):
    """
    Creates the S1 reconstruction of an image. This is a helper function of
    reconstruct_S1_features() with the same parameter meaning.

    Returns:
        A pair consisting of the name under which to write the image and the
        reconstructed image.
    """
    print('Reconstructing S1 features')
    vis_img = np.zeros(target_img.shape)
    vis_parts = visualization_parts(target_img.shape, layer_collection['S1'],
                                    feature_imgs_dict, args.delta)
    for img_pairs in vis_parts.values():
        for img, feature_label in img_pairs:
            vis_img += img
    img_name = 'S1_reconstructions/{}_S1_reconstruction.png'.format(\
                                    plb.Path(args.target_name).stem)
    return (img_name, vis_img)

def reconstruct_S1_features(target_img, layer_collection, feature_imgs_dict,
                            args):
    """
    Reconstructs the recognized features of the S1 layer by drawing
    the features onto a black canvas with their intensity proportional to
    the recognition strength.

    Parameters:
        
        `target_img`: The target image

        `layer_collection`: A dictionary containing for each layer name a
                            dictionary containing for each size a list of layers
                            for all features

        `feature_imgs_dict`: A dictionary containing for each name the
                             corresponding feature image

        `args`: The commandline arguments object. Uses the delta and the target
                image name from it
        
    """
    img_name, vis_img = create_S1_feature_image(target_img, layer_collection,
                                                feature_imgs_dict, args)
    cv2.imwrite(img_name, vis_img)
    
def reconstruct_C1_features(target_img, layer_collection, feature_imgs_dict,
                            args):
    """
    Reconstructs the recognized features of the C1 layer by drawing
    the features onto a copy of the target_img, with their intensity
    proportional to the recognition strength.

    Arguments:

        `target_img`: The target image

        `layer_collection`: A dictionary containing for each layer name a list
                            of those layers

        `feature_imgs_dict`: A dictionary containing for each name the
                             corresponding feature image

        `args`: The commandline arguments object. Uses the delta and the target
                image name from it

    """
    print('Reconstructing C1 features')
    # Create the RGB canvas to draw colored rectangles for the features
    canvas = cv2.cvtColor(target_img, cv2.COLOR_GRAY2RGB)
    # Create the colored squares for the features in a map
    colored_squares_dict = {} # feature name -> colored square
    # A set of predefined colors
    colors = {'red': (255, 0, 0),
              'green': (0, 255, 0),
              'blue': (0, 0, 255),
              'yellow': (255, 255, 0),
              'purple': (255, 0, 255)}
    color_iterator = colors.__iter__()
    for feature_name, feature_img in feature_imgs_dict.items():
        color_name = color_iterator.__next__()
        f_n, f_m = feature_img.shape
        bf_n = 6 * args.delta + f_m   # "big" f_n
        bf_m = 6 * args.delta + f_m   # "big" f_m
        # Create a square to cover all pixels of a C1 neuron
        square = np.zeros((bf_n, bf_m, 3))
        cv2.rectangle(square, (0, 0), (bf_n - 1, bf_m - 1), colors[color_name])
        print('feature name and color: ', feature_name, color_name)
        colored_squares_dict[feature_name] = square
    vis_parts = visualization_parts(target_img.shape, layer_collection['C1'],
                                    colored_squares_dict,
                                    6 * args.delta, canvas)
    img_name_stem = plb.Path(args.target_name).stem
    output_dir = plb.Path(args.c1_output + '/' + img_name_stem)
    if not output_dir.exists():
        output_dir.mkdir()
    for size, img_pairs in vis_parts.items():
        for img, feature_label in img_pairs:
            cv2.imwrite(output_dir.as_posix() + '/{}_{}_C1_{}_reconstruction.png'.\
                        format(img_name_stem, size, feature_label), img)

def reconstruct_S2_features(weights_dicts: Sequence[Dict[str, np.array]],
                            feature_imgs_dict: Dict[str, np.array],
                            f_s: int) -> np.array:
    """
    Reconstructs the weights of the S2 prototype neurons using the features
    passed feature images

    Parameters:
        `weights_dicts`: A list of dictionaries containing for each prototype a
                         the feature name and the connection weights from the S2
                         layer to the C1 layer of that feature

        `filtered_imgs_dict`: A dictionary containing for each feature name an
                              image of the feature

        `f_s`: The size of the recognized features in C1 neurons
    """
    # Determine the dimensions of the output canvas
    s2_prototype_cells = len(weights_dicts)
    n = int(np.sqrt(s2_prototype_cells))
    while n > 1 and s2_prototype_cells % n != 0:
        n -= 1
    m = s2_prototype_cells // n
    f_side_length = f_s * 6 + 1
    t_side_length = f_side_length + 3
    big_canvas = np.zeros( (n * t_side_length, m * t_side_length) )
    # Draw the prototype reconstructions
    for prototype in range(s2_prototype_cells):
        # Determine the highest intensity
        weights_dict = weights_dicts[prototype]
        max_weight = max([max(layer_weights.ravel())\
                            for layer_weights in weights_dict.values()])
        canvas = np.zeros( (f_side_length, f_side_length) )
        for label, weights in weights_dict.items():
            for i in range(len(weights)):
                copy_to_visualization(i, weights[i][0] / max_weight,
                                      feature_imgs_dict[label], canvas,
                                      (f_s, f_s), 6)
        boxed_canvas = 70 * np.ones((t_side_length, t_side_length))
        for i in range(f_side_length):
            for j in range(f_side_length):
                boxed_canvas[i + 1][j + 1] = canvas[i][j]
        copy_to_visualization(prototype, 1, boxed_canvas, big_canvas, (n, m),
                              t_side_length, overfull=False)
    return big_canvas

def plot_C1_spikes(C1_layers: Dict[float, Sequence[nw.Layer]], image_name: str,
                   clear=False, out_dir_name='plots/C1') -> None:
    """
    Plots the spikes of the layers in the given dictionary


    Arguments:

        `C1_layers`: The C1 layers

        `image_name`: The name of the image that will be written. This string
                      will be a part of the actual plot file name.

        `out_dir_name`: The directory where to save the plots
    """
    spike_panels = []
    for size, layers in C1_layers.items():
        spike_panels = []
        for layer in layers:
            out_data = layer.population.get_data(clear=clear).segments[0]
            spike_panels.append(plt.Panel(out_data.spiketrains, xticks=True,
                                          yticks=True,
                                          xlabel='{}, {} scale layer'.format(\
                                                layer.population.label, size)))
        plt.Figure(*spike_panels).save('{}/C1_{}_{}_scale.png'.format(\
                                                out_dir_name, image_name, size))

def plot_S2_spikes(S2_layers: Dict[float, Sequence[nw.Layer]], image_name: str,
                   s2_prototype_cells: int, out_dir_name='plots/S2')\
        -> None:
    """
    Plots the S2 spikes

    Parameters:
        `S2_layers`: A dictionary with the S2 layers

        `image_name`: The name of the image that will be written. This string
                      will be a part of the actual plot file name.

        `s2_prototype_cells`: The number of S2 prototype cells

        `out_dir_name`: The top level directory where to save the plots
    """
    for i in range(s2_prototype_cells):
        spike_panels = []
        for size, layer_list in S2_layers.items():
            layer = layer_list[i]
            out_data = layer.population.get_data().segments[0]
            spike_panels.append(plt.Panel(out_data.spiketrains,
                              xticks=True, yticks=True,
              xlabel='{}, scale {}, prototype {}'.format(image_name, size, i)))
            spike_panels.append(plt.Panel(out_data.filter(name='v')[0],
                              xticks=True, yticks=True,
              xlabel='{}, scale {}, prototype {}'.format(image_name, size, i)))
        plt.Figure(*spike_panels).save('{}/{}/S2_{}_prototype{}.png'.format(\
                                            out_dir_name, i, image_name, i))

def plot_C2_spikes(C2_populations, it, sim_time, plot_name,
                   out_dir_name='plots/C2/'):
    """
    Plots the C2 spikes

    Parameters:
        `C2_populations`: A list containing for each prototype a population of
                          one neuron

        `out_dir_name`: The directory where to save the plot

        `plot_name`: The name of the image
    """
    fig_settings = {
        'lines.linewidth': 0.5,
        'axes.linewidth': 0.5,
        'axes.labelsize': 'small',
        'legend.fontsize': 'small',
    }
    mplt.rcParams.update(fig_settings)
    mplt.figure(figsize=(10, 6))
    start_time = it * sim_time
    mplt.subplot(211)
    mplt.axis([start_time, start_time + sim_time, -.2, len(C2_populations) - .8])
    mplt.xlabel('Time (ms)')
    mplt.ylabel('Neuron index')
    mplt.grid(True)
    for i in range(len(C2_populations)):
        st = C2_populations[i].get_data().segments[0].spiketrains[0]
        mplt.plot(st, np.ones_like(st) * i, '.')
    mplt.subplot(212)
    mplt.axis([start_time, start_time + sim_time, -66, -49])
    for i in range(len(C2_populations)):
        v = C2_populations[i].get_data().segments[0].filter(name='v')[0]
        mplt.plot(v.times, v, label=str(i))

    mplt.savefig('{}/{}.png'.format(out_dir_name, plot_name))
