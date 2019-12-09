
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

from typing import Dict, Sequence, List, Tuple
import numpy as np
import pyNN.nest as sim
import nest
import pyNN.space as space
import pyNN.random as rnd
import argparse as ap
import cv2
import pathlib as plb
import time
import common as cm
import statistics as st

class Layer:
    """
    Represents a layer in the network architecture.

    Attributes:

        `population`: The pyNN neuron population of the layer

        `shape`:      The shape of the layer as a tuple of (rows, cols)

        `projections`: A dictionary containing for each feature name a list of
                       projections of this layer for that feature
    """

    def __init__(self, population, shape):
        self.population = population
        self.shape = shape
        self.current_spike_counts = [0] * population.size
        self.old_spike_counts = [0] * population.size
        self.projections = {} # Dict[str, Sequence[sim.Projection]]

    def update_spike_counts(self):
        """
        Updates the spike counts inside self.current_spike_counts to reflect the
        latest simulation advancement. That is, to store the spike counts from
        the previous simulation to the current one.
        """
        spike_counts = self.population.get_spike_counts()
        for i in range(self.population.size):
            self.current_spike_counts[i] =\
                spike_counts[self.population[i]] - self.old_spike_counts[i]
            self.old_spike_counts[i] = spike_counts[self.population[i]]

def set_i_offsets(layer, source_np_array):
    """
    Sets the i_offset for the input neuons according to the pixel values of the
    input image.

    Parameters:
        `layer`: The layer for which to set the i_offsets

        `source_np_array`: The array with the pixel intensities from which to
                           set the i_offsets
    """
    layer.population.set(i_offset=list(map(lambda x: x / 255 * .6 + .75,
                                           source_np_array.ravel())))

def set_spike_source_layer_rates(layer, source_np_array):
    """
    Sets the firing rates of the already created spike source layer to the
    given source_np_array. This is also a helper function of
    create_spike_source_layer_from().
    """
    layer.population.set(rate=list(map(lambda x: x / 4,
                                       source_np_array.ravel())))

def create_empty_spike_source_layer_with_shape(shape):
    """
    Creates a spike source layer with the given shape and its firing rates set
    to zero. This is also a helper function of create_spike_source_layer_from().
    """
    spike_source_layer = sim.Population(size=shape[0] * shape[1],
                                   cellclass=sim.SpikeSourcePoisson(rate=0))
    return Layer(spike_source_layer, shape)

def create_spike_source_layer_from(source_np_array):
    """
    For a given image returns a layer of spike source neurons to encode the
    image intensities in spikes. The size of the spike source layer is the
    number of pixels in the image.
    """
    layer = create_empty_spike_source_layer_with_shape(source_np_array.shape)
    set_spike_source_layer_rates(layer, source_np_array)
    return layer

def recognizer_weights_from(feature_np_array):
    """
    Builds a network from the firing rates of the given feature_np_array for the
    input neurons and learns the weights to recognize the image through STDP.
    """
    in_p = create_spike_source_layer_from(feature_np_array).population
    out_p = sim.Population(1, sim.IF_curr_exp(i_offset=5))
    synapse = sim.STDPMechanism(weight=-0.2,
              timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,
                                                  A_plus=0.01,
                                                  A_minus=0.005),
              weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.4))
    proj = sim.Projection(in_p, out_p, sim.AllToAllConnector(), synapse)
    sim.run(500)
    return proj.get('weight', 'array')

def connect_layers(input_layer, output_layer, weights, i_s, j_s, i_e, j_e,
                   k_out, stdp=False, initial_weight=0, label_dicts=None):
    """
    Connects a neuron of an output layer to the corresponding square of an input
    layer. This is a helper function of connect_layer_to_layer()

    Returns:
        The created projection between the input and output layers
    """
    m = input_layer.shape[1]
    view_elements = []
    i = i_s
    while i < i_e:
        j = j_s
        while j < j_e:
            view_elements.append(m * i + j)
            j += 1
        i += 1

    if stdp:
        w_max = initial_weight * 15
        stdp_shared = sim.native_synapse_type('stdp_synapse')\
                       (Wmax=w_max * 1000, mu_plus=0.0, mu_minus=1.0)
        proj = sim.Projection(input_layer.population[view_elements],
                              output_layer.population[[k_out]],
                              sim.AllToAllConnector(), stdp_shared)
        ol = int(output_layer.population.label)
        il = input_layer.population.label
        out_neuron = output_layer.population[k_out]
        if label_dicts == None:
            for i in range(len(view_elements)):
                label = '{}_{}_{}'.format(ol, il, i)
                in_neuron = input_layer.population[view_elements[i]]
                conn = nest.GetConnections(source=[in_neuron],
                                           target=[out_neuron])
                nest.SetStatus(conn, {'label': label, 'weight': weights[i][0]})
        else:
            for i in range(len(view_elements)):
                label = '{}_{}_{}'.format(ol, il, i)
                if not label in label_dicts[ol]:
                    label_dicts[ol][label] = ([], [])
                in_neuron = input_layer.population[view_elements[i]]
                label_dicts[ol][label][0].append(in_neuron)
                label_dicts[ol][label][1].append(out_neuron)
    else:
        proj = sim.Projection(input_layer.population[view_elements],
                              output_layer.population[[k_out]],
                              sim.AllToAllConnector(),
                              sim.StaticSynapse(weight=weights))
    return proj

def how_many_squares_in_shape(input_shape, feature_shape, delta):
    """
    Computes the shape of an output layer that can be plugged into the input
    layer with respect to their shapes and the delta

    Parameters;
        `input_shape`: The shape of the input layer

        `feature_shape`: The shape of the feature

        `delta`: The vertical and horizontal offset of the output layers squares

    Returns:
        A pair representing the shape of an output layer that can be plugged
        into the input layer with respect to their shapes and the delta
    """
    # Determine how many output neurons can be connected to the input layer
    # according to the deltas
    t_n, t_m = input_shape
    f_n, f_m = feature_shape
    if t_n < f_n or t_m < f_m:
        raise Exception('Feature shape {} is greater than layer shape {}'.format(\
                                                   feature_shape, input_shape))
    n = int((t_n - f_n) / delta) + ((t_n - f_n) % delta > 0) + 1
    m = int((t_m - f_m) / delta) + ((t_m - f_m) % delta > 0) + 1
    return (n, m)

def connect_layer_to_layer(input_layer, output_layer, feature_shape, delta,
                           weights, stdp=False, initial_weight=0,
                           ndicts=None, ondicts=None, omdicts=None)\
        -> List[sim.Projection]:
    """
    Connects a full input layer to a full output layer by connecting each
    neuron of the output layer to a square of neurons in the input layer
    according to the shape of the square and the delta.

    Parameters:
        `input_layer`: The input layer

        `output_layer`: The output layer

        `feature_shape`: The shape of the squares of the input which will be
                         connected to one output neuron

        `delta`: The vertical and horizontal offset of the output layers squares

        `weights`: A list of weights in the shape returned by
                   Projection.get('weight') 

        `ndicts`, `ondicts`, `omdicts`: Lists of dictionaries to store the
        edges for non-overlapping, overlapping in n and
        overlapping in m squares respectively. 

    Returns:
        A list of the projections of the input layer to the output layer.
    """
    # Go through the lines of the image and connect input neurons to the
    # output layer according to delta
    t_n, t_m = input_layer.shape
    f_n, f_m = feature_shape
    overfull_n = (t_n - f_n) % delta > 0 # True for vertical overflow
    overfull_m = (t_m - f_m) % delta > 0 # True for horizontal overflow
    k_out = 0
    projections = []
    i = 0
    while i + f_n <= t_n:
        j = 0
        while j + f_m <= t_m:
            projections.append(connect_layers(input_layer, output_layer,
                                              weights, i, j, i + f_n, j + f_m,
                                              k_out, stdp=stdp,
                                              initial_weight=initial_weight,
                                              label_dicts=ndicts))
            k_out += 1
            j += delta
        if overfull_m:
            projections.append(connect_layers(input_layer, output_layer,
                                              weights, i, t_m - f_m, i + f_n,
                                              t_m, k_out, stdp=stdp,
                                              initial_weight=initial_weight,
                                              label_dicts=omdicts))
            k_out += 1
        i += delta
    if overfull_n:
        j = 0
        while j + f_m <= t_m:
            projections.append(connect_layers(input_layer, output_layer,
                                              weights, t_n - f_n, j, t_n,
                                              j + f_m, k_out, stdp=stdp,
                                              initial_weight=initial_weight,
                                              label_dicts=ondicts))
            k_out += 1
            j += delta
        if overfull_m:
            projections.append(connect_layers(input_layer, output_layer,
                                              weights, t_n - f_n, t_m - f_m,
                                              t_n, t_m, k_out, stdp=stdp,
                                              initial_weight=initial_weight,
                                              label_dicts=None))
            k_out += 1
    return projections
    

def create_output_layer(input_layer, weights_tuple, delta, layer_name, refrac):
    """
    Builds a layer which connects to the input_layer according to the given
    parameters.

    Parameters:
        `input_layer`: The input layer

        `weights_tuple`: A tuple of the form (weights, weights_shape)

        `delta`: The vertical and horizontal offset of the output layers squares
        
        `layer_name`: The name of the input layer

        `refrac`: The refractory period of the output layer neurons

    Returns:
        An output layer which is connected to the given input layer according
        to the given parameters
    """
#    print('Number of output neurons {} for size {}x{}'.format(\
#                                            total_output_neurons, t_n, t_m))
    n, m = how_many_squares_in_shape(input_layer.shape, weights_tuple[1], delta)
    total_output_neurons = n * m
    print('Layer:', layer_name)
    print('Output layer has shape', n, m)
    output_layer = Layer(sim.Population(total_output_neurons,
                                       sim.IF_curr_exp(tau_refrac=refrac),
                                       structure=space.Grid2D(aspect_ratio=m/n),
                                       label=layer_name), (n, m))
    connect_layer_to_layer(input_layer, output_layer, weights_tuple[1], delta,
                           weights_tuple[0])

    return output_layer

def create_corner_layer_for(input_layers):
    shape = input_layers[0].shape
    total_output_neurons = np.prod(shape)

    output_population = sim.Population(total_output_neurons, sim.IF_curr_exp(),
                                       label='corner')
    for layer in input_layers:
        sim.Projection(layer.population,
                       output_population,
                       sim.OneToOneConnector(),
                       sim.StaticSynapse(weight=1., delay=0.5))

    return Layer(output_population, shape)

def train_weights(feature_dir):
    """
    Trains the basic recognizer weights such that they respond to the features
    found in the directory feature_dir. This function runs a sim.start() -
    sim.end() "session".

    Arguments:

        `feature_dir`: The directory where the features are stored as images

    Returns:

        A pair of weight and feature image dictionaries of the following type:
            weights_dict        :: feature name string -> (weights, shape)
            feature_imgs_dict   :: feature name string -> feature image
    """
    sim.setup()
    weights_dict = {}       # feature name string -> (weights, shape)
    feature_imgs_dict = {}  # feature name string -> feature image
    for training_img in plb.Path(feature_dir).iterdir():
        feature_np_array = cv2.imread(training_img.as_posix(), cv2.CV_8U)
        feature_imgs_dict[training_img.stem] = feature_np_array
        weights = recognizer_weights_from(feature_np_array)
        weights_dict[training_img.stem] = (weights, feature_np_array.shape)
    sim.end()
    return (weights_dict, feature_imgs_dict)

def change_rates_for_scales(input_layers, target):
    """
    Sets the rates of the given input layers according to the values of the
    target image.
    """
    for size, layer in input_layers.items():
        resized_target = cv2.resize(src=target, dsize=None,
                                    fx=size, fy=size,
                                    interpolation=cv2.INTER_AREA)
        set_spike_source_layer_rates(layer, resized_target)

def create_input_layers_for_scales(target, scales, is_bag=False):
    """
    Creates for a given target image and a list of scales a list of input spike
    layers one for each size.

    Parameters:

        `target`: The target image or stream from which to create the input
                  layers

        `scales`: A list of scales for which to create input layers

        `is_bag`: If set to True, the passed target will be treated as a rosbag,
                  otherwise as an image

    Returns:

        A list of input spike layers created from the given target, one for each
        scale
    """
    input_layers = {}
    bag_input_layers = {}
    t_n = target.shape[0]
    t_m = target.shape[1]
    for size in scales:
        if is_bag:
            resized_target = stream.resize_stream(target, size)
            bag_input_layers[size] =\
                 create_spike_source_layer_from_stream(resized_target)
        else:
            t1 = time.clock()
            input_layer = create_empty_spike_source_layer_with_shape(\
                                       ( round(t_n * size), round(t_m * size) ))
            print('Input layer creation for scale {} took {} s'.format(size,
                                                             time.clock() - t1))
            print('Input layer for scale {} has {} neurons'.format(size,
                                   input_layer.shape[0] * input_layer.shape[1]))
            input_layers[size] = input_layer
    if is_bag:
        return bag_input_layers
    change_rates_for_scales(input_layers, target)
    return input_layers

def set_i_offsets_for_all_scales_to(layers_dict: Dict[float, Sequence[Layer]],
                                    target: np.array) -> None:
    """
    Sets the i_offsets of the layers to the intensities of the gabor filtered
    target image.

    Parameters:
        `layers_dict`: A dictionary containing for each size a list of layers,
                       for each gabor filter orientation one.
                  
        `target`: The unfiltered image to create the gabor convoltions from
    """
    for size, layers in layers_dict.items():
        print("size",size)
        resized_target = cv2.resize(src=target, dsize=None, fx=size, fy=size,
                                    interpolation=cv2.INTER_AREA)
        resized_target.shape
        filtered_imgs_dict = cm.get_gabor_edges(resized_target)
        for layer in layers:
            set_i_offsets(layer, filtered_imgs_dict[layer.population.label])

def set_blank_i_offsets(layers_dict: Dict[float, Sequence[Layer]]) -> None:
    """
    Sets the i_offsets of the given layers to zero.

    Parameters:
        `layers_dict`: A dictionary containing for each size a list of layers,
                       for each gabor filter orientation one.
    """
    for layers in layers_dict.values():
        for layer in layers:
            layer.population.set(i_offset=0)

def create_empty_input_layers_for_scales(target: np.array, scales: [float])\
        -> Dict[float, List[Layer]]:
    """
    Creates empty input layers from the target image shape for every scale in
    the passed list.

    Parameters:
        `target`: The target image from which to compute the gabor filters and
                  create the input layers

        `scales`: A list of the scales for which to create input layers
    
    Returns:
        A dictionary which contains for each scale a list of four input layers,
        one for each orientation
    """
    input_layers = {}
    feature_names = cm.get_gabor_feature_names()
    for size in scales:
        print('Creating input layers for size', size)
        n = round(target.shape[0] * size)
        m = round(target.shape[1] * size)
        input_layers[size] = [Layer(sim.Population(n * m,
                                                   sim.IF_curr_exp(),
                                label=feature_name), (n, m))\
                              for feature_name in feature_names] 
    return input_layers
    
def create_gabor_input_layers_for_scales(target: np.array, scales: [float])\
        -> Dict[float, List[Layer]]:
    """
    Creates input layers from the given image by using gabor filters in four
    orientations. The firing rate of the neurons is controlled by the i_offsets
    which are set according to the convolution intensities.

    Parameters:
        `target`: The target image from which to compute the gabor filters and
                  create the input layers

        `scales`: A list of the scales for which to create input layers

    Returns:
        A dictionary which contains for each scale a list of four input layers,
        one for each orientation
    """
    input_layers = create_empty_input_layers_for_scales(target, scales)
    set_i_offsets_for_all_scales_to(input_layers, target)
    return input_layers

def create_S1_layers(input_layers_dict, weights_dict, args):
    """
    Creates S1 layers for the given input layers. It creates for each input
    layer a S1 layer for each feature in the weights_dict

    Parameters:

        `input_layers_dict`: A dictionary of input layers for which to create
                             the S1 layers for all features in the weights_dict.
                             It stores for each size the corresponding input
                             layer

        `weights_dict`: A dictionary containing for each feature name a pair
                        of a weight list and its shape

        `args`: The commandline arguments

    Returns:

        A dictionary containing for each size of the target a list of S1 layers
    """
    S1_layers = {} # input size -> list of S1 feature layers
    for size, input_layer in input_layers_dict.items():
        print('Create S1 layers for size', size)
        t1 = time.clock()
        S1_layers[size] = [create_output_layer(input_layer, weights_tuple,
                                               args.delta, layer_name,
                                               args.refrac_s1)\
                          for layer_name, weights_tuple in weights_dict.items()]
        print('S1 layer creation for scale {} took {} s'.format(size,
                                                            time.clock() - t1))

        neuron_count = sum(map(lambda layer: layer.shape[0] * layer.shape[1],
                               S1_layers[size]))
        print('S1 layers at scale {} have {} neurons'.format(size, neuron_count))
    return S1_layers

def create_cross_layer_inhibition(layers_dict):
    """
    Creates inhibitory connections between the given feature layers for each
    size to allow only the spikes of the strongest feature to be propagated
    further

    Parameters:
        
        `layers_dict`: A dictionary of layers of the type size -> list of layers
    """
    def inhibitory_connect(layers, source, dest1, dest2, dest3, weight):
        sim.Projection(layers[source].population, layers[dest1].population,
                       sim.OneToOneConnector(), sim.StaticSynapse(weight=weight))
        sim.Projection(layers[source].population, layers[dest2].population,
                       sim.OneToOneConnector(), sim.StaticSynapse(weight=weight))
        sim.Projection(layers[source].population, layers[dest3].population,
                       sim.OneToOneConnector(), sim.StaticSynapse(weight=weight))

    for size, layers in layers_dict.items():
        print('Create cross layer inhibiton for size', size)
        inhibitory_connect(layers, 0, 1, 2, 3, -50)
        inhibitory_connect(layers, 1, 0, 2, 3, -50)
        inhibitory_connect(layers, 2, 0, 1, 3, -50)
        inhibitory_connect(layers, 3, 0, 1, 2, -50)

def create_C1_layers(S1_layers_dict: Dict[float, Sequence[Layer]],
                     refrac_c1: float) -> Dict[float, Sequence[Layer]]:
    """
    Creates C1 layers for each of the given S1 layers

    Arguments:

        `S1_layers_dict`: A dictionary containing for each size of the input
                          image a list of S1 layers, for each feature one

        `refrac_c1`:

    Returns:

        A dictionary containing for each size of S1 layers a list of C1 layers
    """
    C1_layers = {} # input size -> list of C1 layers
    for size, S1_layers in S1_layers_dict.items():
        C1_subsampling_shape = (7, 7)
        neuron_number = C1_subsampling_shape[0] * C1_subsampling_shape[1]
        move = 6
        C1_weight = .35
        weights_tuple = (C1_weight * np.ones((neuron_number, 1)),
                         C1_subsampling_shape)
        t1 = time.clock()
        C1_layers[size] = list(map(lambda S1_layer:\
                                    create_output_layer(S1_layer, weights_tuple,
                                    move, S1_layer.population.label, refrac_c1),
                                   S1_layers))
        print('C1 layer creation for scale {} took {} s'.format(size,
                                                            time.clock() - t1))
        neuron_count = sum(map(lambda layer: layer.shape[0] * layer.shape[1],
                               C1_layers[size]))
        print('C1 layers at scale {} have {} neurons'.format(size, neuron_count))
    return C1_layers

def create_local_inhibition(layers_dict):
    """
    Creates local inhibitory connections from a neuron to its neighbors in an
    area of a fixed distance. The latency of its neighboring neurons decreases
    linearly with the distance from the spike from 15% to 5%, as described in
    Masquelier's paper. Here we assumed that a weight of -10 inhibits the
    neuron completely and took that as a starting point.
    """
    for size, layers in layers_dict.items():
        print('Create local inhibition for size', size)
        for layer in layers:
            sim.Projection(layer.population, layer.population,
                sim.DistanceDependentProbabilityConnector('d < 5',
                    allow_self_connections=False),
                sim.StaticSynapse(weight='.25 * d - 1.75'),
                space=space.Space(axes='xy')) 

def initialize_label_dicts(s2_prototype_cells: int, f_s: int)\
        -> List[Dict[str, Tuple[List[int], List[int]]]]:
    """
    Initializes the label dict with the corresponding labels as keys

    Parameters:
        `s2_prototype_cells`: The number of S2 prototype cells

        `f_s`: The feature size in C1 neurons

    Returns:
        A dictionary containing the labels as keys mapping to pairs of empty
        lists
    """
    s2_label_dicts = [None] * s2_prototype_cells
    neurons = f_s * f_s
    for prototype in range(s2_prototype_cells):
        s2_label_dicts[prototype] = []
        for label in cm.get_gabor_feature_names():
            for i in range(neurons):
                s2_label_dicts[prototype]['{}_{}_{}'\
                    .format(prototype, label, i)] = ([], [])
    return s2_label_dicts

def create_S2_layers(C1_layers: Dict[float, Sequence[Layer]], feature_size,
                     s2_prototype_cells, refrac_s2=.1, stdp=True,
                     inhibition=True)\
        -> Dict[float, List[Layer]]:
    """
    Creates all prototype S2 layers for all sizes.

    Parameters:
        `layers_dict`: A dictionary containing for each size a list of C1
                       layers, for each feature one

        `feature_size`:

        `s2_prototype_cells`:

        `refrac_s2`:

        `stdp`: 

    Returns:
        A dictionary containing for each size a list of different S2
        layers, for each prototype one.
    """
    f_s = feature_size
    initial_weight = 25 / (f_s * f_s)
    weight_rng = rnd.RandomDistribution('normal', mu=initial_weight,
                                                  sigma=initial_weight / 20)
    i_offset_rng = rnd.RandomDistribution('normal', mu=.5, sigma=.45)
    weights = list(map(lambda x: weight_rng.next() * 1000, range(4 * f_s * f_s)))
    S2_layers = {}
    i_offsets = list(map(lambda x: i_offset_rng.next(),
                     range(s2_prototype_cells)))
    ndicts = list(map(lambda x: {}, range(s2_prototype_cells)))
    ondicts = list(map(lambda x: {}, range(s2_prototype_cells)))
    omdicts = list(map(lambda x: {}, range(s2_prototype_cells)))
    for size, layers in C1_layers.items():
        n, m = how_many_squares_in_shape(layers[0].shape, (f_s, f_s), f_s)
        if stdp:
            l_i_offsets = [list(map(lambda x: rnd.RandomDistribution('normal',
                             mu=i_offsets[i], sigma=.25).next(), range(n * m)))\
                                for i in range(s2_prototype_cells)]
        else:
            l_i_offsets = np.zeros((s2_prototype_cells, n * m))
        print('S2 Shape', n, m)
        layer_list = list(map(lambda i: Layer(sim.Population(n * m,
                                     sim.IF_curr_exp(i_offset=l_i_offsets[i],
                                                     tau_refrac=refrac_s2),
                                     structure=space.Grid2D(aspect_ratio=m/n),
                                     label=str(i)), (n, m)),
                              range(s2_prototype_cells)))
        for S2_layer in layer_list:
            for C1_layer in layers:
                S2_layer.projections[C1_layer.population.label] =\
                    connect_layer_to_layer(C1_layer, S2_layer, (f_s, f_s), f_s,
                                           [[w] for w in weights[:f_s * f_s]],
                                           stdp=stdp,
                                           initial_weight=initial_weight,
                                           ndicts=ndicts, ondicts=ondicts,
                                           omdicts=omdicts)
        S2_layers[size] = layer_list
    # Set the labels of the shared connections
    if stdp:
        t = time.clock()
        print('Set shared labels')
        for s2_label_dicts in [ndicts, ondicts, omdicts]:
            for i in range(s2_prototype_cells):
                w_iter = weights.__iter__()
                for label, (source, target) in s2_label_dicts[i].items():
                    conns = nest.GetConnections(source=source, target=target)
                    nest.SetStatus(conns, {'label': label,
                                          'weight': w_iter.__next__()})
    print('Setting labels took', time.clock() - t)
    if inhibition:
        # Create inhibitory connections between the S2 cells
        # First between the neurons of the same layer...
        inh_weight = -10
        inh_delay = .1
        print('Create S2 self inhibitory connections')
        for layer_list in S2_layers.values():
            for layer in layer_list:
                sim.Projection(layer.population, layer.population,
                               sim.AllToAllConnector(allow_self_connections=False),
                               sim.StaticSynapse(weight=inh_weight,
                                                 delay=inh_delay))
        # ...and between the layers
        print('Create S2 cross-scale inhibitory connections')
        for i in range(s2_prototype_cells):
            for layer_list1 in S2_layers.values():
                for layer_list2 in S2_layers.values():
                    if layer_list1[i] != layer_list2[i]:
                        sim.Projection(layer_list1[i].population,
                                       layer_list2[i].population,
                                       sim.AllToAllConnector(),
                                       sim.StaticSynapse(weight=inh_weight,
                                                         delay=inh_delay))
    if stdp:
        # Create the inhibition between different prototype layers
        print('Create S2 cross-prototype inhibitory connections')
        for layer_list in S2_layers.values():
            for layer1 in layer_list:
                for layer2 in layer_list:
                    if layer1 != layer2:
                        sim.Projection(layer1.population, layer2.population,
                                       sim.OneToOneConnector(),
                                       sim.StaticSynapse(weight=inh_weight-1,
                                                         delay=inh_delay))
    return S2_layers

def set_s2_weights(S2_layers: Dict[float, Sequence[Layer]], prototype: int,
                   active_layer=None, first_neuron=0,
                   weights_dict_list=None) -> None:
    """
    Set the weights of a prototype S2 layer either to the weights of a specific
    neuron in the given layer or to a given dict of weights

    Parameters:
        `S2_layers`: A dictionary containing for each size the corresponding S2
                     layer

        `prototype`: The prototype index for which to copy the weights

        `active_layer`: The layer in which the neuron first_neuron contains the
                        weights to be copied to the S2_layers

        `first_neuron`: The neuron in active_layer from which to copy the
                        weights to the S2_layers
                        
        `weights_dict_list`: A list containing for each prototype a dictionary
                             containing for each feature label a weights array
                             to be copied to the S2_layers
    """
    for layer_list in S2_layers.values():
        current_layer = layer_list[prototype]
        for label, projections in current_layer.projections.items():
            for proj in projections:
                if weights_dict_list == None:
                    proj.set(weight=active_layer.projections[label][first_neuron]\
                                                        .get('weight', 'array'))
                else:
                    proj.set(weight=weights_dict_list[prototype][label])

def update_shared_weights(S2_layers: Dict[float, Sequence[Layer]],
                          s2_prototype_cells: int) -> List[Dict[str, np.array]]:
    """
    Updates the weights of the "shared" projections of the S2 neurons.

    Parameters:
        `S2_layers`: A dictionary containing for each size the corresponding S2
                     layer

        `s2_prototype_cells`: The number of S2 prototype cells

    Returns:
        A list containing for each prototype the weights of the connections of
        the neuron which fired first in the respective prototype.
    """
    weights_dict_list = []
    for i in range(s2_prototype_cells):
        earliest_spike = np.infty
        first_neuron = 0
        active_layer = None
        # Determine the neuron that fired first and the layer it is in
        for layer_list in S2_layers.values():
            current_layer = layer_list[i]
            current_spiketrains = current_layer.population.get_data(clear=True)\
                                                          .segments[0].spiketrains
            for j in range(len(current_spiketrains)):
                if len(current_spiketrains[j]) > 0\
                        and current_spiketrains[j][0] < earliest_spike:
                    print(current_spiketrains[j][0])
                    earliest_spike = current_spiketrains[j][0]
                    first_neuron = j
                    active_layer = current_layer

        # Copy the weights of the neuron in the active layer to all other S2 layers
        if earliest_spike < np.infty:
            set_s2_weights(S2_layers, active_layer=active_layer,
                           first_neuron=first_neuron, prototype=i)
            weights_dict_list.append(\
                dict([(label, projections[first_neuron].get('weight', 'array'))\
                    for label, projections in active_layer.projections.items()]))
        # If no earliest spike was found, for e.g. if no neuron fired, take the
        # weights of any neuron from the population, e.g. the first one.
        else:
            weights_dict_list.append(\
                dict([(label, projections[first_neuron].get('weight', 'array'))\
                for label, projections in\
                    list(S2_layers.values())[0][i].projections.items()]))
    return weights_dict_list

def get_current_weights(S2_layers: Dict[float, Sequence[Layer]],
                        s2_prototype_cells: int) -> List[Dict[str, np.array]]:
    """
    Gets the weights of the "shared" projections of the S2 neurons.

    Parameters:
        `S2_layers`: A dictionary containing for each size the corresponding S2
                     layer

        `s2_prototype_cells`: The number of S2 prototype cells

    Returns:
        A list containing for each prototype the weights of the connections of
        the neuron which fired first in the respective prototype.
    """
    weights_dict_list = []
    for i in range(s2_prototype_cells):
        weights_dict = dict([(label, projections[0].get('weight', 'array'))\
                    for label, projections in\
                        list(S2_layers.values())[0][i].projections.items()])
        mean = st.mean(np.array(list(weights_dict.values())).ravel())
        for weights_list in weights_dict.values():
            for k in range(len(weights_list)):
                if weights_list[k][0] < mean:
                    weights_list[k][0] = 0
        weights_dict_list.append(weights_dict)
    return weights_dict_list

def create_C2_layers(S2_layers: Dict[float, Sequence[Layer]],
                     s2_prototype_cells: int) -> List[sim.Population]:
    """
    Creates the populations of the C2 layer, one for each S2 prototype cell,
    containing only a single cell which max-pools the spikes of all layers of a
    prototype.

    Parameters:
        `S2_layers`: A dictionary containing for each scale a list of S2
                     layers, one for each prototype cell

        `s2_prototype_cells`: The number of S2 prototype cells

    Returns:
        A list of populations of size one, one population for each prototype
        cell
    """
    no_inh_w = 17.15            # synapse weight without S2 inhibitions
    with_inh_w = 4 * no_inh_w   # synapse weight with S2 inhibitions
    C2_populations = [sim.Population(1, sim.IF_curr_exp(),
                                     label=str(prot))\
                        for prot in range(s2_prototype_cells)]
    total_connections = sum(map(lambda ll: ll[0].shape[0] * ll[0].shape[1],
                                S2_layers.values()))
    for s2ll in S2_layers.values():
        for prot in range(s2_prototype_cells):

            sim.Projection(s2ll[prot].population, C2_populations[prot],
                           sim.AllToAllConnector(),
                           sim.StaticSynapse(weight=with_inh_w / total_connections))
    return C2_populations
