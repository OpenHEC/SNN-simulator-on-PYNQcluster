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
import sys
sys.path.append('/home/xilinx/nest/lib/python3.6/site-packages')
import numpy as np
import cv2
import pyNN.nest as sim
import pathlib as plb
import time
import pickle
import argparse as ap
import re

import network as nw

parser = ap.ArgumentParser('./dump-c2-spikes.py --')
parser.add_argument('--training-c1-dumpfile', type=str, required=True,
                    help='The output file to contain the C1 spiketrains for\
                         training')
parser.add_argument('--validation-c1-dumpfile', type=str, required=True,
                    help='The output file to contain the C1 spiketrains for\
                         validation')
parser.add_argument('--threads', default=1, type=int)
parser.add_argument('--weights-from', type=str, required=True,
                    help='Dumpfile of the S2 weight array')
args = parser.parse_args()

sim.setup(threads=args.threads, min_delay=.1)

layer_collection = {}

# Extracting meta-information about the simulation from the filename
training_dumpfile_name = plb.Path(args.training_c1_dumpfile).stem
validation_dumpfile_name = plb.Path(args.validation_c1_dumpfile).stem
training_image_count = int(re.search('\d*imgs',
                                     training_dumpfile_name).group()[:-4])
validation_image_count = int(re.search('\d*imgs',
                                       validation_dumpfile_name).group()[:-4])
training_sim_time = float(re.search('\d+\.\d+ms',
                                        training_dumpfile_name).group()[:-2])
validation_sim_time = float(re.search('\d+\.\d+ms',
                                        validation_dumpfile_name).group()[:-2])
blanktime = 0
occurrence = re.search('\d+\.\d+blank', training_dumpfile_name)
if occurrence is not None:
    blanktime = float(occurrence.group()[:-5])

print('Create C1 layers')
t1 = time.clock()
training_ddict = pickle.load(open(args.training_c1_dumpfile, 'rb'))
validation_ddict = pickle.load(open(args.validation_c1_dumpfile, 'rb'))
layer_collection['C1'] = {}
for size, layers_as_dicts in training_ddict.items():
    layer_list = []
    for layer_as_dict in layers_as_dicts:
        n, m = layer_as_dict['shape']
        new_layer = nw.Layer(sim.Population(n * m,
                        sim.SpikeSourceArray(),
                        label=layer_as_dict['label']), (n, m))
        layer_list.append(new_layer)
    layer_collection['C1'][size] = layer_list
print('C1 creation took {} s'.format(time.clock() - t1))

print('Creating S2 layers and reading the epoch weights')
epoch_weights_list = pickle.load(open(args.weights_from, 'rb'))
epoch = epoch_weights_list[-1][0]
weights_dict_list = epoch_weights_list[-1][1]
f_s = int(np.sqrt(list(weights_dict_list[0].values())[0].shape[0]))
s2_prototype_cells = len(weights_dict_list)
layer_collection['S2'] = nw.create_S2_layers(layer_collection['C1'], f_s,
                                             s2_prototype_cells, refrac_s2=.1,
                                             stdp=False, inhibition=True)

print('Creating C2 layers')
t1 = time.clock()
layer_collection['C2'] = nw.create_C2_layers(layer_collection['S2'],
                                             s2_prototype_cells)
print('C2 creation took {} s'.format(time.clock() - t1))

for pop in layer_collection['C2']:
    pop.record('spikes')

def set_c1_spiketrains(ddict):
    for size, layers_as_dicts in ddict.items():
        for layer_as_dict in layers_as_dicts:
            spiketrains = layer_as_dict['segment'].spiketrains
            dimensionless_sts = [[s for s in st] for st in spiketrains]
            the_layer_iter = filter(lambda layer: layer.population.label\
                            == layer_as_dict['label'], layer_collection['C1'][size])
            the_layer_iter.__next__().population.set(spike_times=dimensionless_sts)

def extract_spiketrains(image_count, sim_time):
    print('========= Start simulation =========')
    print('Simulating for', image_count, 'images')
    sim.run((sim_time + blanktime) * image_count)
    print('========= Stop  simulation =========')
    return [layer_collection['C2'][prot].get_data(clear=True).segments[0]\
                .spiketrains[0] for prot in range(s2_prototype_cells)]

c2_training_spikes = []
c2_validation_spikes = []

for epoch, weights_dict_list in epoch_weights_list:
    # Set the S2 weights to those from the file
    print('Setting S2 weights to epoch', epoch)
    for prototype in range(s2_prototype_cells):
        nw.set_s2_weights(layer_collection['S2'], prototype,
                          weights_dict_list=weights_dict_list)

    print('Setting C1 spike trains to the training dataset')
    set_c1_spiketrains(training_ddict)
    # Let the simulation run to "fill" the layer pipeline with spikes
    sim.run(40)
    print('>>>>>>>>> Extracting spike trains for learning <<<<<<<<<')
    c2_training_spikes.append((epoch,
                  extract_spiketrains(training_image_count, training_sim_time)))
    sim.reset()

    print('Setting C1 spike trains to the validation dataset')
    set_c1_spiketrains(validation_ddict)
    # Let the simulation run to "fill" the layer pipeline with spikes
    sim.run(40)
    print('>>>>>>>>> Extracting spike trains for validation <<<<<<<<<')
    c2_validation_spikes.append((epoch,
              extract_spiketrains(validation_image_count, validation_sim_time)))
    sim.reset()

c2_training_dumpfile_name = 'C2_spikes/{}_fs{}_{}prots.bin'\
                      .format(training_dumpfile_name, f_s, s2_prototype_cells)
c2_validation_dumpfile_name = 'C2_spikes/{}_fs{}_{}prots.bin'\
                      .format(validation_dumpfile_name, f_s, s2_prototype_cells)
c2_training_dumpfile = open(c2_training_dumpfile_name, 'wb')
c2_validation_dumpfile = open(c2_validation_dumpfile_name, 'wb')
print('Dumping C2 training spikes to file', c2_training_dumpfile_name)
print('Dumping C2 validation spikes to file', c2_validation_dumpfile_name)
pickle.dump(c2_training_spikes, c2_training_dumpfile, protocol=4)
pickle.dump(c2_validation_spikes, c2_validation_dumpfile, protocol=4)

sim.end()
