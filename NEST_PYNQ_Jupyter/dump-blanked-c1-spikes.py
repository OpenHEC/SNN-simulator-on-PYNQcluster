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
import pickle
import argparse as ap

import network as nw
import visualization as vis

parser = ap.ArgumentParser('./dump-blanked-c1-spikes.py --')
parser.add_argument('--blanktime', type=float, default=120,
                    help='The blank time between the image spikes. Defaults to\
                    120 ms')
parser.add_argument('--dataset-label', type=str, required=True,
                    help='The name of the dataset which was used for\
                    training')
parser.add_argument('--training-dir', type=str, required=True,
                    help='The directory with the training images')
parser.add_argument('--refrac-c1', type=float, default=.1, metavar='0.1',
                    help='The refractory period of neurons in the C1 layer in\
                    ms')
parser.add_argument('--sim-time', default=50, type=float, help='Simulation time',
                    metavar='50')
parser.add_argument('--scales', default=[1.0, 0.71, 0.5, 0.35],
                    nargs='+', type=float,
                    help='A list of image scales for which to create\
                    layers. Defaults to [1, 0.71, 0.5, 0.35]')
parser.add_argument('--threads', default=1, type=int)
args = parser.parse_args()

training_path = plb.Path(args.training_dir)
imgs = [(filename.stem, cv2.imread(filename.as_posix(), cv2.CV_8UC1))\
            for filename in sorted(training_path.glob('*.png'))]

sim.setup(threads=args.threads)

layer_collection = {}

print('Create S1 layers')
t1 = time.clock()
layer_collection['S1'] =\
    nw.create_gabor_input_layers_for_scales(imgs[0][1], args.scales)
nw.create_cross_layer_inhibition(layer_collection['S1'])
print('S1 layer creation took {} s'.format(time.clock() - t1))

print('Create C1 layers')
t1 = time.clock()
layer_collection['C1'] = nw.create_C1_layers(layer_collection['S1'],
                                             args.refrac_c1)
nw.create_local_inhibition(layer_collection['C1'])
print('C1 creation took {} s'.format(time.clock() - t1))

for layer_name in ['C1']:
    if layer_name in layer_collection:
        for layers in layer_collection[layer_name].values():
            for layer in layers:
                layer.population.record('spikes')

# Simulate for a certain time to allow the whole layer pipeline to "get filled"
sim.run(40)

print('========= Start simulation =========')
start_time = time.clock()
count = 0
for filename, target_img in imgs:
    t1 = time.clock()
    print('Simulating for', filename, 'number', count)
    count += 1
    nw.set_i_offsets_for_all_scales_to(layer_collection['S1'], target_img)
    sim.run(args.sim_time)
    nw.set_blank_i_offsets(layer_collection['S1'])
    sim.run(args.blanktime)
    print('Took', time.clock() - t1, 'seconds')
end_time = time.clock()
print('========= Stop  simulation =========')
print('Simulation took', end_time - start_time, 's')

ddict = {}
dataset_label = '{}_{}imgs_{}ms_{}px_scales'.format(args.dataset_label,
                                   len(imgs), args.sim_time,
                                   imgs[0][1].shape[0])
for size, layers in layer_collection['C1'].items():
    ddict[size] = [{'segment': layer.population.get_data().segments[0],
                    'shape': layer.shape,
                    'label': layer.population.label } for layer in layers]
    dataset_label += '_{}'.format(size)
dataset_label += '_{}blank'.format(float(args.blanktime))

dumpname = 'C1_spikes/{}.bin'.format(dataset_label)
print('Dumping spikes for all scales and layers to file', dumpname)
dumpfile = open(dumpname, 'wb')
pickle.dump(ddict, dumpfile, protocol=4)
dumpfile.close()

sim.end()
