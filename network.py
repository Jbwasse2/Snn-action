import matplotlib.pyplot as plt
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


# Create SNN
def build_SNN(image_size, config):
    with nengo.Network(seed=config["seed"]) as net:
        # remove some unnecessary features to speed up the training
        nengo_dl.configure_settings(
            trainable=True, stateful=True, keep_history=True,
        )

        # input node
        inp = nengo.Node(np.zeros(image_size[-1]))

        # lmu cell
        lmu = LMUCell(units=800, order=2, theta=image_size[1], input_d=image_size[-1])
        conn = nengo.Connection(inp, lmu.x, synapse=None)
        net.config[conn].trainable = False

        # dense linear readout
        out = nengo.Node(size_in=101)
        nengo.Connection(lmu.h, out, transform=nengo_dl.dists.Glorot(), synapse=None)

        # record output. note that we set keep_history=False above, so this will
        # only record the output on the last timestep (which is all we need
        # on this task)
        p = nengo.Probe(out)
    return net
