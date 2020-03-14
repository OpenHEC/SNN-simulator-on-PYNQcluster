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

try:
    from mpi4py import MPI
except ImportError:
    raise Exception("Trying to gather data without MPI installed. If you are\
    not running a distributed simulation, this is a bug in PyNN.")

parser = ap.ArgumentParser('./dump-single-c1-spikes.py --')
parser.add_argument('--refrac-c1', type=float, default=.1, metavar='0.1',
                    help='The refractory period of neurons in the C1 layer in\
                    ms')
parser.add_argument('--sim-time', default=50, type=float, help='Simulation time',
                    metavar='50')
parser.add_argument('--scales', default=[1.0, 0.71, 0.5, 0.35, 0.25],
                    nargs='+', type=float,
                    help='A list of image scales for which to create\
                    layers. Defaults to [1, 0.71, 0.5, 0.35, 0.25]')
parser.add_argument('--target-name', type=str,
                    help='The name of the already edge-filtered image to be\
                    recognized')
parser.add_argument('--threads', default=1, type=int)
args = parser.parse_args()

MPI_ROOT = 0

def is_root():
    return MPI.COMM_WORLD.rank == MPI_ROOT 

target_path = plb.Path(args.target_name)
target_img = cv2.imread(target_path.as_posix(), cv2.CV_8UC1)

sim.setup(threads=args.threads)

layer_collection = {}

print('Create S1 layers')
t1 = time.clock()
layer_collection['S1'] =\
    nw.create_gabor_input_layers_for_scales(target_img, args.scales)
nw.create_cross_layer_inhibition(layer_collection['S1'])
print('S1 layer creation took {} s'.format(time.clock() - t1))

print('Create C1 layers')
t1 = time.clock()
layer_collection['C1'] = nw.create_C1_layers(layer_collection['S1'],
                                             args.refrac_c1)
nw.create_local_inhibition(layer_collection['C1'])
print('C1 creation took {} s'.format(time.clock() - t1))

for layers in layer_collection['C1'].values():
    for layer in layers:
        layer.population.record('spikes')

print('========= Start simulation =========')
start_time = time.clock()
sim.run(args.sim_time)
end_time = time.clock()
print('========= Stop  simulation =========')
print('Simulation took', end_time - start_time, 's')

print('Dumping spikes for all scales and layers')
ddict = {}
filename = 'C1_spike_data/' + target_path.stem
for size, layers in layer_collection['C1'].items():
    ddict[size] = [{'segment': layer.population.get_data().segments[0],
                    'shape': layer.shape,
                    'label': layer.population.label } for layer in layers]
    filename += '_{}'.format(size)
if is_root():
    dumpfile = open('{}_{}ms_norefrac.bin'.format(filename, args.sim_time), 'wb')
    pickle.dump(ddict, dumpfile, protocol=4)
    dumpfile.close()

sim.end()
