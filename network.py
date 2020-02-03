from collections import deque

import matplotlib.pyplot as plt
from nengo.utils.numpy import is_iterable
import random
import nengo
import nengo_dl
import nengo_loihi
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from nengo.utils.filter_design import cont2discrete


# Create LMU cell
# See https://www.nengo.ai/nengo-dl/examples/lmu.html
class LMUCell(nengo.Network):
    def __init__(self, units, order, theta, input_d, **kwargs):
        super().__init__(**kwargs)
        # compute the A and B matrices according to the LMU's mathematical derivation
        # (see the paper for details)
        Q = np.arange(order, dtype=np.float64)
        R = (2 * Q + 1)[:, None] / theta
        j, i = np.meshgrid(Q, Q)

        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
        C = np.ones((1, order))
        D = np.zeros((1,))

        A, B, _, _, _ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")

        with self:
            nengo_dl.configure_settings(trainable=None)

            # create objects corresponding to the x/u/m/h variables in the above diagram
            self.x = nengo.Node(size_in=input_d)
            self.u = nengo.Node(size_in=1)
            self.m = nengo.Node(size_in=order)
            self.h = nengo_dl.TensorNode(tf.nn.tanh, shape_in=(units,), pass_time=False)

            # compute u_t from the above diagram.
            # note that setting synapse=0 (versus synapse=None) adds a one-timestep
            # delay, so we can think of any connections with synapse=0 as representing
            # value_{t-1}
            nengo.Connection(
                self.x, self.u, transform=np.ones((1, input_d)), synapse=None
            )
            nengo.Connection(self.h, self.u, transform=np.zeros((1, units)), synapse=0)
            nengo.Connection(self.m, self.u, transform=np.zeros((1, order)), synapse=0)

            # compute m_t
            # in this implementation we'll make A and B non-trainable, but they
            # could also be optimized in the same way as the other parameters
            conn_A = nengo.Connection(self.m, self.m, transform=A, synapse=0)
            self.config[conn_A].trainable = False
            conn_B = nengo.Connection(self.u, self.m, transform=B, synapse=None)
            self.config[conn_B].trainable = False

            # compute h_t
            nengo.Connection(
                self.x, self.h, transform=np.zeros((units, input_d)), synapse=None
            )
            nengo.Connection(
                self.h, self.h, transform=np.zeros((units, units)), synapse=0
            )
            nengo.Connection(
                self.m,
                self.h,
                transform=nengo_dl.dists.Glorot(distribution="normal"),
                synapse=None,
            )


class LMUCellSpike(nengo.Network):
    def __init__(self, units, order, theta, input_d, **kwargs):
        super().__init__(**kwargs)

        # compute the A and B matrices according to the LMU's mathematical derivation
        # (see the paper for details)
        Q = np.arange(order, dtype=np.float64)
        R = (2 * Q + 1)[:, None] / theta
        j, i = np.meshgrid(Q, Q)

        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
        C = np.ones((1, order))
        D = np.zeros((1,))

        A, B, _, _, _ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")

        with self:
            nengo_dl.configure_settings(trainable=None)

            # create objects corresponding to the x/u/m/h variables in the above diagram
            self.h = nengo_dl.TensorNode(tf.nn.tanh, shape_in=(units,), pass_time=False)
            n_ensembles = 2
            self.x = nengo.networks.EnsembleArray(
                n_neurons=100,
                n_ensembles=n_ensembles,
                ens_dimensions=input_d,
                neuron_type=nengo.LIF(),
            )
            self.u = nengo.networks.EnsembleArray(
                n_neurons=100,
                n_ensembles=n_ensembles,
                ens_dimensions=1,
                neuron_type=nengo.LIF(),
            )
            self.m = nengo.networks.EnsembleArray(
                n_neurons=100,
                n_ensembles=n_ensembles,
                ens_dimensions=order,
                neuron_type=nengo.LIF(),
            )

            # compute u_t from th=20above diagram.
            # note that setting synapse=0 (versus synapse=None) adds a one-timestep
            # delay, so we can think of any connections with synapse=0 as representing
            # value_{t-1}
            nengo.Connection(
                self.x.output, self.u.input, synapse=None, transform=nengo_dl.dists.Glorot(distribution="normal")
            )
            nengo.Connection(self.h, self.u.input, transform=nengo_dl.dists.Glorot(distribution="normal"), synapse=0)
            nengo.Connection(self.m.output, self.u.input, transform=nengo_dl.dists.Glorot(distribution="normal"), synapse=0)

            # compute m_t
            # in this implementation we'll make A and B non-trainable, but they
            # could also be optimized in the same way as the other parameters
            conn_A = nengo.Connection(self.m.output, self.m.input, transform=nengo_dl.dists.Glorot(distribution="normal"), synapse=0)
            self.config[conn_A].trainable = True
            conn_B = nengo.Connection(self.u.output, self.m.input, transform=nengo_dl.dists.Glorot(distribution="normal"), synapse=None)
            self.config[conn_B].trainable = True

            # compute h_t
            nengo.Connection(
                self.x.output, self.h, transform=nengo_dl.dists.Glorot(distribution="normal"), synapse=None
            )
            nengo.Connection(
                self.h, self.h, transform=np.zeros((units, units)), synapse=0
            )
            nengo.Connection(
                self.m.output,
                self.h,
                transform=nengo_dl.dists.Glorot(distribution="normal"),
                synapse=None,
            )

# Create SNN
def build_SNN(image_size, config, spiking=True):
    with nengo.Network(seed=config["seed"]) as net:
        # remove some unnecessary features to speed up the training
        nengo_dl.configure_settings(stateful=False)

        # input node
        inp = nengo.Node(np.zeros(image_size[-1]))
        out = nengo.Node(size_in=101)

        if spiking:
            # lmu cell
            lmu = LMUCellSpike(
                units=800, order=20, theta=image_size[1], input_d=image_size[-1]
            )
            conn = nengo.Connection(inp, lmu.x.input,transform = nengo_dl.dists.Glorot(distribution="normal"), synapse=None)
        else:
            lmu = LMUCell(
                units=800, order=2, theta=image_size[1], input_d=image_size[-1]
            )
            conn = nengo.Connection(inp, lmu.x, synapse=None)
            # dense linear readout

        nengo.Connection(lmu.h, out, transform=nengo_dl.dists.Glorot(), synapse=None)
        p = nengo.Probe(out)
    return net

def build_SNN_simple(image_size, config):
    with nengo.Network(seed=config["seed"]) as net:
        # remove some unnecessary features to speed up the training
        nengo_dl.configure_settings(stateful=False)
        n_ensembles = 1
        ens_dimension = 1
        recurrent_size = 10000
        depth = 1
        # input node

        inp = nengo.Node(np.zeros(image_size[-1]))  #
        out = nengo.Node(size_in=101)
        ensembles_in = []
        ensembles_out = []
        #Create ensembles
        for i in range(depth):
#            u1 = nengo.networks.EnsembleArray(
#                n_neurons=1,
#                n_ensembles=n_ensembles,
#                ens_dimensions=ens_dimension,
#                neuron_type=nengo.SpikingRectifiedLinear(),
#            )

            u2 = RecurrentEnsembleArray(
                n_neurons=1,
                n_ensembles=recurrent_size,
                ens_dimensions=1,
                timing_length=29,
                recurrent_connection_percentage=0.01,
                neuron_type=nengo.SpikingRectifiedLinear()
                )

#            out_u1 = nengo.Node(size_in=recurrent_size)
#            nengo.Connection(u1.output, u2.input, transform=nengo_dl.dists.Glorot(), synapse=None)
            ensembles_in.append(u2.input)
            ensembles_out.append(u2.output)
        #Connect middle ensembles
        if depth != 1:
            for i in range(1,depth-1):
                curr_in = ensembles_in[i]
                post_in = ensembles_in[i+1]
                pre_out = ensembles_out[i-1]
                curr_out = ensembles_out[i]

                nengo.Connection(pre_out, curr_in)
                nengo.Connection(curr_out, post_in)
            #Connect first ensemble
            nengo.Connection(inp, ensembles_in[0], transform=nengo_dl.dists.Glorot())
            nengo.Connection(ensembles_in[0], ensembles_out[1])
            #Connect last ensemble
            nengo.Connection(ensembles_out[-1], out, transform=nengo_dl.dists.Glorot(), synapse=None)
            nengo.Connection(ensembles_out[-2], ensembles_in[-1])
        else:
            nengo.Connection(inp, ensembles_in[0], transform=nengo_dl.dists.Glorot())
            nengo.Connection(ensembles_out[0], out, transform=nengo_dl.dists.Glorot(), synapse=None)
        p = nengo.Probe(out)
    return net



class RecurrentEnsembleArray(nengo.networks.EnsembleArray, nengo.Network ):
    def __init__(
                   self,
        n_neurons,
        n_ensembles,
        ens_dimensions=1,
        label=None,
        seed=None,
        add_to_container=None,
        timing_length=29,
        recurrent_connection_percentage=0.10,
        **ens_kwargs 
            ):
        nengo.Network.__init__(self, label, seed, add_to_container)
        for param in ens_kwargs:
            if is_iterable(ens_kwargs[param]):
                ens_kwargs[param] = nengo.dists.Samples(ens_kwargs[param])

        self.config[nengo.Ensemble].update(ens_kwargs)

        label_prefix = "" if label is None else label + "_"

        self.n_neurons_per_ensemble = n_neurons
        self.n_ensembles = n_ensembles
        self.dimensions_per_ensemble = ens_dimensions

        # These may be set in add_neuron_input and add_neuron_output
        self.neuron_input, self.neuron_output = None, None

        self.ea_ensembles = []
        #Used for dropout
        #Key = label id (just number, no prefix), values = the ensemble and its connections
        self.conns = {}

        with self:
            self.input = nengo.Node(size_in=self.dimensions, label="input")

            for i in range(n_ensembles):
                ensemble_conn = []
                e = nengo.Ensemble(
                    n_neurons,
                    self.dimensions_per_ensemble,
                    label="%s%d" % (label_prefix, i),
                )
                # Make the node non-passthrough
                # This is useful for dropout
                self.input.output =  lambda t, x: x
                #Create recurrent connections
                if random.uniform(0, 1) < recurrent_connection_percentage:
                    timing = random.uniform(0,timing_length)
                    #0.001 is simulation step time
                    conn_r = nengo.Connection(e, e, transform=nengo_dl.dists.Glorot(), synapse=timing*0.001)
                    ensemble_conn.append(conn_r)
                conn = nengo.Connection(
                    self.input[i * ens_dimensions : (i + 1) * ens_dimensions],
                    e,
                    synapse=None,
                )
                ensemble_conn.append(conn)
                d = {i:ensemble_conn}
                self.conns.update(d)
                self.ea_ensembles.append(e)

            self.add_output("output", function=None)
