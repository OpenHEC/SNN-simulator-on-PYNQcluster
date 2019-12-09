#!/bin/ipython

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

import numpy as np
import cv2
import sys
import pyNN.nest as sim
import pathlib as plb
import time

import common as cm
import network as nw
import visualization as vis
import time

args = cm.parse_args()

t1 = time.clock()
weights_dict, feature_imgs_dict = nw.train_weights(args.feature_dir)
print('Training weights took {} s'.format(time.clock() - t1))

if args.plot_weights:
    vis.plot_weights(weights_dict)
    sys.exit(0)

# The training part is done. Go on with the "actual" simulation
sim.setup(threads=4)

target_img = cm.read_and_prepare_img(args.target_name, args.filter)
if args.filter != 'none':
    filename = 'edges/{}_{}_edges.png'.format(plb.Path(args.target_name).stem,
                                              args.filter)
    if not plb.Path(filename).exists():
        cv2.imwrite(filename, target_img)

layer_collection = {} # layer name -> dict of S1 layers of type
                      # 'scale -> layer list'
layer_collection['input'] = nw.create_input_layers_for_scales(target_img,
                                                             args.scales)
t1 = time.clock()
layer_collection['S1'] = nw.create_S1_layers(layer_collection['input'],
                                             weights_dict, args)
nw.create_cross_layer_inhibition(layer_collection['S1'])
print('S1 creation took {} s'.format(time.clock() - t1))

if not args.no_c1:
    t1 = time.clock()
    print('Create C1 layers')
    layer_collection['C1'] = nw.create_C1_layers(layer_collection['S1'],
                                                 args.refrac_c1)
    nw.create_local_inhibition(layer_collection['C1'])
    print('C1 creation took {} s'.format(time.clock() - t1))

for layer_name in ['S1', 'C1']:
    if layer_name in layer_collection:
        for layers in layer_collection[layer_name].values():
            for layer in layers:
                layer.population.record('spikes')

print('========= Start simulation =========')
start_time = time.clock()
sim.run(args.sim_time)
end_time = time.clock()
print('========= Stop  simulation =========')
print('Simulation took', end_time - start_time, 's')

for layer_name in ['S1', 'C1']:
    if layer_name in layer_collection:
        for layers in layer_collection[layer_name].values():
            for layer in layers:
                layer.update_spike_counts()

t1 = time.clock()
if args.reconstruct_s1_img:
    vis.reconstruct_S1_features(target_img, layer_collection, feature_imgs_dict,
                                args)
    print('S1 visualization took {} s'.format(time.clock() - t1))

t1 = time.clock()
if args.reconstruct_c1_img and not args.no_c1:
    vis.reconstruct_C1_features(target_img, layer_collection, feature_imgs_dict,
                                args)
    print('C1 visualization took {} s'.format(time.clock() - t1))


t1 = time.clock()
if args.plot_spikes:
    print('Plotting spikes')
    vis.plot_spikes(layer_collection, args)
    print('Plotting spiketrains took {} s'.format(time.clock() - t1))

sim.end()
