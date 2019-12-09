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
import re

import common as cm
import network as nw
import visualization as vis
import time

parser = ap.ArgumentParser('./learn-features.py --')
parser.add_argument('--c1-dumpfile', type=str, required=True,
                    help='The output file to contain the C1 spiketrains')
parser.add_argument('--epoch-size', type=int, default=30,
                    help='The lenght of an epoch')
parser.add_argument('--feature-size', type=int, default=3,
                     help='The size of the features to be learnt')
parser.add_argument('--s2-prototype-cells', type=int, default=3,
                    help='The number of S2 features to compute')
parser.add_argument('--plot-c1-spikes', action='store_true',
                    help='Plot the spike trains of the C1 layers')
parser.add_argument('--plot-s2-spikes', action='store_true',
                    help='Plot the spike trains of the S2 layers')
#parser.add_argument('--refrac-s2', type=float, default=.1, metavar='.1',
#                    help='The refractory period of neurons in the S2 layer in\
#                    ms')
parser.add_argument('--threads', default=1, type=int)
args = parser.parse_args()

sim.setup(threads=args.threads, min_delay=.1)

layer_collection = {}

# Read the gabor features for reconstruction
feature_imgs_dict = {} # feature string -> image
for filepath in plb.Path('features_gabor').iterdir():
    feature_imgs_dict[filepath.stem] = cv2.imread(filepath.as_posix(),
                                                  cv2.CV_8UC1)

# Extracting meta-information about the simulation from the filename
c1_dumpfile_name = plb.Path(args.c1_dumpfile).stem
image_count = int(re.search('\d*imgs', c1_dumpfile_name).group()[:-4])
#sim_time = float(re.search('\d+\.\d+ms', c1_dumpfile_name).group()[:-2])
sim_time = float(50)
dataset_label = '{}_fs{}_{}prots'.format(c1_dumpfile_name, args.feature_size,
                                         args.s2_prototype_cells)

print('Create C1 layers')
t1 = time.clock()
dumpfile = open(args.c1_dumpfile, 'rb')
ddict = pickle.load(dumpfile)
layer_collection['C1'] = {}
for size, layers_as_dicts in ddict.items():
    layer_list = []
    for layer_as_dict in layers_as_dicts:
        n, m = layer_as_dict['shape']
        spiketrains = layer_as_dict['segment'].spiketrains
        dimensionless_sts = [[s for s in st] for st in spiketrains]
        new_layer = nw.Layer(sim.Population(n * m,
                        sim.SpikeSourceArray(spike_times=dimensionless_sts),
                        label=layer_as_dict['label']), (n, m))
        layer_list.append(new_layer)
    layer_collection['C1'][size] = layer_list
print('C1 creation took {} s'.format(time.clock() - t1))

print('Creating S2 layers')
t1 = time.clock()
layer_collection['S2'] = nw.create_S2_layers(layer_collection['C1'],
                                             args.feature_size,
                                             args.s2_prototype_cells,
                                             refrac_s2=6)
print('S2 creation took {} s'.format(time.clock() - t1))

#for layers in layer_collection['C1'].values():
#    for layer in layers:
#        layer.population.record('spikes')
#for layer_list in layer_collection['S2'].values():
#    for layer in layer_list:
#        layer.population.record(['spikes', 'v'])

reconstructions_dir_dataset_path = plb.Path('S2_reconstructions/' + dataset_label)
if not reconstructions_dir_dataset_path.exists():
    reconstructions_dir_dataset_path.mkdir(parents=True)
if args.plot_c1_spikes:
    c1_plots_dir_path = plb.Path('plots/C1/' + dataset_label)
    if not c1_plots_dir_path.exists():
        c1_plots_dir_path.mkdir(parents=True)
if args.plot_s2_spikes:
    s2_plots_dataset_dir = plb.Path('plots/S2/' + dataset_label)
    for i in range(args.s2_prototype_cells):
        s2_plots_dir_path = s2_plots_dataset_dir / str(i)
        if not s2_plots_dir_path.exists():
            s2_plots_dir_path.mkdir(parents=True)

epoch_weights = [] # type: List[Tuple[int, List[Dict[str, np.array]]]]

# Let the simulation run to "fill" the layer pipeline with spikes
sim.run(40)

print('========= Start simulation =========')
start_time = time.clock()
for i in range(image_count):
    print('Simulating for image number', i)
    sim.run(sim_time)
    if args.plot_c1_spikes:
        vis.plot_C1_spikes(layer_collection['C1'],
                           '{}_image_{}'.format(dataset_label, i),
                           out_dir_name=c1_plots_dir_path.as_posix())
    if args.plot_s2_spikes:
        vis.plot_S2_spikes(layer_collection['S2'],
                       '{}_image_{}'.format(dataset_label, i),
                       args.s2_prototype_cells,
                       out_dir_name=s2_plots_dataset_dir.as_posix())
    if (i + 1) % 10 == 0:
        current_weights = nw.get_current_weights(layer_collection['S2'],
                                                 args.s2_prototype_cells)
        cv2.imwrite('{}/{}_{:0>4}_images.png'.format(\
                 reconstructions_dir_dataset_path.as_posix(),
                 dataset_label, i + 1),
            vis.reconstruct_S2_features(current_weights,
                                        feature_imgs_dict,
                                        args.feature_size))
    if (i + 1) % args.epoch_size == 0:
        current_weights = nw.get_current_weights(layer_collection['S2'],
                                                 args.s2_prototype_cells)
        epoch_weights.append((i + 1, current_weights))
end_time = time.clock()
print('========= Stop  simulation =========')
print('Simulation took', end_time - start_time, 's')
# Reconstruct the last image
cv2.imwrite('{}/{}_{:0>4}_images.png'.format(\
         reconstructions_dir_dataset_path.as_posix(),
         dataset_label, i + 1),
    vis.reconstruct_S2_features(current_weights,
                                feature_imgs_dict,
                                args.feature_size))
# Also add the weights of the last iteration to the dumpfile
if image_count % args.epoch_size != 0:
    epoch_weights.append((image_count, current_weights))

dumpfile_name = 'S2_weights/{}.bin'.format(dataset_label)
out_dumpfile = open(dumpfile_name, 'wb')
print('Dumping weights for the selected epochs to file', dumpfile_name)
pickle.dump(epoch_weights, out_dumpfile, protocol=4)
out_dumpfile.close()

sim.end()
