
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

import pathlib as plb
import numpy as np
import pickle
import argparse as ap
import re
import time
import matplotlib.pyplot as mplt
import pyNN.nest as sim
import statistics as st
from sklearn import metrics

parser = ap.ArgumentParser('./classify-images-one-shot.py --')
parser.add_argument('--training-c2-dumpfile', type=str, required=True,
                    help='The output file to contain the C2 spiketrains for\
                         training')
parser.add_argument('--validation-c2-dumpfile', type=str, required=True,
                    help='The output file to contain the C2 spiketrains for\
                         validation')
parser.add_argument('--training-labels', type=str, required=True,
                    help='Text file which contains the labels of the training\
                          dataset')
parser.add_argument('--validation-labels', type=str, required=True,
                    help='Text file which contains the labels of the validation\
                          dataset')
parser.add_argument('--threads', default=1, type=int)
args = parser.parse_args()

# Get metadata from filenames
training_dumpfile_name = plb.Path(args.training_c2_dumpfile).stem
validation_dumpfile_name = plb.Path(args.validation_c2_dumpfile).stem
training_image_count = int(re.search('\d*imgs',
                                     training_dumpfile_name).group()[:-4])
validation_image_count = int(re.search('\d*imgs',
                                       validation_dumpfile_name).group()[:-4])
training_sim_time = float(re.search('\d*ms',
                                       training_dumpfile_name).group()[:-2])
validation_sim_time = float(re.search('\d*ms',
                                        validation_dumpfile_name).group()[:-2])
#training_sim_time=float(50)
#validation_sim_time=float(50)
#imgs_per_category = int(re.search('\d+learn',
#                                        training_dumpfile_name).group()[:-5])
imgs_per_category=1
blanktime = 0
occurrence = re.search('\d+\.\d+blank', training_dumpfile_name)
if occurrence is not None:
    blanktime = float(occurrence.group()[:-5])
categories = training_image_count // imgs_per_category

# Read the training and validation labels from file
training_labels = open(args.training_labels, 'r').read().splitlines()
validation_labels = open(args.validation_labels, 'r').read().splitlines()

# Read the spike train structures from the pickled dumpfiles
c2_training_spikes = pickle.load(open(args.training_c2_dumpfile, 'rb'))
c2_validation_spikes = pickle.load(open(args.validation_c2_dumpfile, 'rb'))
s2_prototype_cells = len(c2_training_spikes[0][1])
t1=time.clock()
def create_C2_populations(spiketrains):
    C2_populations = [sim.Population(1,
                            sim.SpikeSourceArray(spike_times=[spiketrains[prot]]),
                            label=str(prot))\
                        for prot in range(len(spiketrains))]
    compound_C2_population = C2_populations[0]
    for pop in C2_populations[1:]:
        compound_C2_population += pop
    return (C2_populations, compound_C2_population)

results_label = '{}_{}valimgs_{}valsimtime'\
                    .format(training_dumpfile_name, validation_image_count,
                            validation_sim_time)
print('create_C2_populations took {} s'.format(time.clock() - t1))
def plot_spikes(C2_populations, classifier_neurons, t_sim_time, appendix):
    fig_settings = {
        'lines.linewidth': 0.5,
        'axes.linewidth': 0.5,
        'axes.labelsize': 'small',
        'legend.fontsize': 'small',
    }
    mplt.rcParams.update(fig_settings)
    mplt.figure(figsize=(13, 10))
    mplt.subplot(311)
    mplt.grid(True)
    mplt.axis([0, t_sim_time, -.2, len(C2_populations) - .8])
    for i in range(len(C2_populations)):
        st = C2_populations[i].get_data().segments[0].spiketrains[0]
        mplt.plot(st, np.ones_like(st) * i, '.')
    mplt.subplot(312)
    mplt.axis([0, t_sim_time, -.2, len(classifier_neurons) - .8])
    for i in range(len(classifier_neurons)):
        st = classifier_neurons[i].get_data().segments[0].spiketrains[0]
        mplt.plot(st, np.ones_like(st) * i, '.')
    mplt.subplot(313)
    mplt.axis([0, t_sim_time, -66, -49])
    for i in range(len(classifier_neurons)):
        segm = classifier_neurons[i].get_data().segments[0]
        voltages = segm.filter(name='v')[0]
        mplt.plot(voltages.times, voltages, label=str(i))
    mplt.savefig('plots/CLF/{}_{}.png'.format(results_label, appendix))

# Datastructure to store the learned weights from all epochs
all_epochs_weights = []
start=time.clock()
for training_pair, validation_pair in\
        zip(c2_training_spikes, c2_validation_spikes):
    print("111111")
    # ============= Training ============= #
    print('Construct training network')
    t1=time.clock()
    sim.setup(threads=args.threads, min_delay=.1)
    # Create the C2 layer and connect it to the single output neuron
    training_spiketrains = [[s for s in st] for st in training_pair[1]]
    C2_populations, compound_C2_population =\
            create_C2_populations(training_spiketrains)
    out_p = sim.Population(1, sim.IF_curr_exp(tau_refrac=.1))
    stdp_weight = 7 / s2_prototype_cells
    stdp = sim.STDPMechanism(weight=stdp_weight,
           timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=26.0,
                                               A_plus=stdp_weight / 5,
                                               A_minus=stdp_weight / 4.48),
           weight_dependence=sim.AdditiveWeightDependence(w_min=0.0,
                                                      w_max=15.8 * stdp_weight))
    learn_proj = sim.Projection(compound_C2_population, out_p,
                                sim.AllToAllConnector(), stdp)

    epoch = training_pair[0]
    print('Simulating for epoch', epoch)

    # Record the spikes for visualization purposes and to count the number of
    # fired spikes
#    compound_C2_population.record('spikes')
    out_p.record(['spikes', 'v'])

    # Let the simulation run to "fill" the layer pipeline with spikes
    sim.run(40)

    # Datastructure for storing the computed STDP weights for this epoch
    classifier_weights = []
    
    # Training STDP weights for every category and save them to classifier_weights
    old_spike_counts = 0
    max_spikes = 18
    for category in range(categories):
        print('Train weights for', training_labels[category])
        max_simtime = training_sim_time + blanktime
        segm_time = 10
        total_time = 0
        while total_time < max_simtime\
                    and list(out_p.get_spike_counts().values())[0]\
                        - old_spike_counts < max_spikes:
            sim.run(segm_time)
            total_time += segm_time
        old_spike_counts = list(out_p.get_spike_counts().values())[0]
        classifier_weights.append(learn_proj.get('weight', 'array'))
        learn_proj.set(weight=0)
        sim.run(max_simtime - total_time)
        learn_proj.set(weight=stdp_weight)
    print("one-shot stdp train",time.clock()-t1)
#    plot_spikes(C2_populations, [out_p],
#                (training_sim_time + blanktime) * training_image_count + 40,
#                simtime + blanktime,
#                'training')

    sim.end()

    # ============= Validation ============= #
    print('Constructing new network with the learned weights')
    t1=time.clock()
    sim.setup(threads=args.threads, min_delay=.1)

    # Create the validation network and connect the C2 neurons to it
    validation_spiketrains = [[s for s in st] for st in validation_pair[1]]
    C2_populations, compound_C2_population =\
                                create_C2_populations(validation_spiketrains)
    classifier_neurons = [sim.Population(1, sim.IF_curr_exp())\
                                for cat in range(categories)]
    for category in range(categories):
#    for category in range(1):
        sim.Projection(compound_C2_population, classifier_neurons[category],
                       sim.AllToAllConnector(),
                       sim.StaticSynapse(weight=classifier_weights[category]))

    # Record the spikes for visualization purposes
    compound_C2_population.record('spikes')
    for pop in classifier_neurons:
        pop.record(['spikes', 'v'])

    # Let the simulation run to "fill" the layer pipeline with spikes
    sim.run(40)

    for pop in C2_populations:
        pop.get_data(clear=True)

    predicted_labels = []
    # Simulate and classify the images
    for i in range(validation_image_count):
        print('Simulating for image', i)
        sim.run(validation_sim_time + blanktime)
        # Find the neuron which fired most
        label, count = max(zip(training_labels,
                       map(lambda pop: list(pop.get_spike_counts().values())[0],
                           classifier_neurons)), key=lambda pair: pair[1])
        predicted_labels.append(label)
        for clf_n in classifier_neurons:
            clf_n.get_data(clear=True)
    print("one-shot stdp validation ",time.clock()-t1)
#    plot_spikes(C2_populations,
#                classifier_neurons,
#                (validation_sim_time + blanktime) * num_images + 40,
#                'validation')

#    logfile = open('log/{}.log'.format(results_label), 'a')
#    print('============================================================',
#          file=logfile)
#    print('Epoch', epoch, file=logfile)
#    logfile.close()
    all_epochs_weights.append((epoch, classifier_weights))
    sim.end()

#    logfile.close()
    all_epochs_weights.append((epoch, classifier_weights))
    sim.end()
print('validation took {} s'.format(time.clock() - start))
"""
    clf_report = metrics.classification_report(validation_labels, predicted_labels)
    conf_matrix = metrics.confusion_matrix(validation_labels, predicted_labels)
    print(clf_report, file=logfile)
    print(clf_report)
    print(conf_matrix, file=logfile)
    print(conf_matrix)
"""
'''
print('create_stdp_populations took {} s'.format(time.clock() - start))
print('Wrote log to file', logfile.name())
clf_dumpname = 'CLF_weights/{}.bin'.format(results_label)
clf_dumpfile = open(clf_dumpname, 'wb', protocol=4)
print('Dumping classificator weights to file', clf_dumpname)
pickle.dump(all_epochs_weights, clf_dumpfile)
'''
