
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
import rosbag
import pathlib as plb

class Stream:
    def __init__(self, events, shape, duration):
        self.events = events
        self.shape = shape
        self.duration = duration

def resize_stream(stream, size):
    # no interpolation so far
    resized_shape = np.ceil(np.multiply(stream.shape, size)).astype(int)
    resized_events = np.copy(stream.events)
    for event in resized_events:
        event.x = int(np.floor(event.x * size))
        event.y = int(np.floor(event.y * size))
    return Stream(resized_events, resized_shape, stream.duration)

def read_stream(filename):
    bag = rosbag.Bag(filename)
    allEvents = []
    initial_time = None
    last_time = 0
    for topic, msg, t in bag.read_messages(topics=['/dvs/events']):
        if not initial_time and msg.events:
            # we want the first event to happen at 1ms
            initial_time = int(msg.events[0].ts.to_sec() * 1000) - 1
        for event in msg.events:
            event.ts = int(event.ts.to_sec() * 1000) - initial_time
        allEvents = np.append(allEvents, msg.events)
        last_time = t.to_sec() * 1000
        # NOTE: I forgot to specify in the Layer class that the shape is
        # specified in matrix notation like (rows, cols). So here maybe
        # (msg.height, msg.width) could be more appropriate?
        shape = [msg.width, msg.height]
    bag.close()

    return Stream(allEvents, shape, last_time - initial_time)
