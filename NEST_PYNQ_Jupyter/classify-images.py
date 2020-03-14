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
import pyNN.nest as sim
import pathlib as plb
import time
import pickle
import argparse as ap
import re
from sklearn import svm, metrics

import network as nw
import visualization as vis

parser = ap.ArgumentParser('./classify-images.py --')
parser.add_argument('--training-c1-dumpfile', type=str, required=True,
                    help='The output file to contain the C1 spiketrains for\
                         training')
parser.add_argument('--validation-c1-dumpfile', type=str, required=True,
                    help='The output file to contain the C1 spiketrains for\
                         validation')
parser.add_argument('--training-labels', type=str, required=True,
                    help='Text file which contains the labels of the training\
                          dataset')
parser.add_argument('--validation-labels', type=str, required=True,
                    help='Text file which contains the labels of the validation\
                          dataset')
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
sim_time = float(re.search('\d+\.\d+ms', validation_dumpfile_name).group()[:-2])
blanktime = 0
occurrence = re.search('\d+\.\d+blank', training_dumpfile_name)
if occurrence is not None:
    blanktime = float(occurrence.group()[:-5])
occurrence = re.search('\d+\.\d+blank', validation_dumpfile_name)
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
                                             stdp=False, inhibition=False)

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

training_labels = open(args.training_labels, 'r').read().splitlines()
validation_labels = open(args.validation_labels, 'r').read().splitlines()

def clear_data(C2_populations):
    for pop in C2_populations:
        pop.get_data(clear=True)

def extract_data_samples(image_count):
    samples = []
    print('========= Start simulation =========')
    for i in range(image_count):
        print('Simulating for image number', i)
        sim.run(sim_time + blanktime)
        spikes =\
            [list(layer_collection['C2'][prot].get_spike_counts().values())[0]\
                for prot in range(s2_prototype_cells)]
        samples.append(spikes)
        clear_data(layer_collection['C2'])
    print('========= Stop  simulation =========')
    return samples

#### Temporary reduction to every third epoch! #############
epoch_weights_list = [index_pair[1] for index_pair in\
                        filter(lambda index_pair: index_pair[0] % 3 == 0,
                            zip(range(len(epoch_weights_list)),
                                epoch_weights_list))]
############################################################

for epoch, weights_dict_list in epoch_weights_list:
    # Set the S2 weights to those from the file
    print('Setting S2 weights to epoch', epoch)
    for prototype in range(s2_prototype_cells):
        nw.set_s2_weights(layer_collection['S2'], prototype,
                          weights_dict_list=weights_dict_list)

    training_samples = []
    validation_samples = []

    print('Setting C1 spike trains to the training dataset')
    set_c1_spiketrains(training_ddict)
    # Let the simulation run to "fill" the layer pipeline with spikes
    sim.run(40)
    clear_data(layer_collection['C2'])
    print('>>>>>>>>> Extracting data samples for fitting <<<<<<<<<')
    training_samples = extract_data_samples(training_image_count)
    sim.reset()

    print('Setting C1 spike trains to the validation dataset')
    set_c1_spiketrains(validation_ddict)
    # Let the simulation run to "fill" the layer pipeline with spikes
    sim.run(40)
    clear_data(layer_collection['C2'])
    print('>>>>>>>>> Extracting data samples for validation <<<<<<<<<')
    validation_samples = extract_data_samples(validation_image_count)
    sim.reset()

    print('Fitting SVM model onto the training samples')

    clf = svm.SVC(kernel='linear')
    clf.fit(training_samples, training_labels)

    logfile = open('log_final/{}.log'.format(plb.Path(args.weights_from).stem), 'a')

    print('Predicting the categories of the validation samples')
    predicted_labels = clf.predict(validation_samples)
    print('============================================================',
          file=logfile)
    print('Epoch', epoch, file=logfile)
    clf_report = metrics.classification_report(validation_labels, predicted_labels)
    conf_matrix = metrics.confusion_matrix(validation_labels, predicted_labels)
    print(clf_report, file=logfile)
    print(clf_report)
    print(conf_matrix, file=logfile)
    print(conf_matrix)

    logfile.close()

sim.end()
