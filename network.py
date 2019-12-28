import numpy as np
import torch
import torch.nn as nn
import nengo
import nengo_loihi
import nengo_dl

#Create Conv Network (Just copy VGG16 or Resnet idc)
model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet101', pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
model.fc = nn.Linear(2048, 2048)

#Create SNN 
def build_model(learning_rate=1e-5, arm_sim_dt=1e-3, f_ext_scale=100.):
    with nengo.Network(label="action_network") as model:
        def gen_target(t):
            #Add data loader class here
            return 0
        target_node = nengo.Node(output=gen_target, size_out=ndim)


        model.arm_node = nengo.Node(output=arm_func, size_in=ndim, size_out=ndim)

        model.probe_target = nengo.Probe(target_node)  # track targets
        # model.probe_pos = nengo.Probe(model.arm_node)  # track hand (x,y)
        def arm_pos(t):
            return theRod.r[1:,:].ravel()

        model.arm_pos = nengo.Node(output=arm_pos)#, size_in=nelem, size_out=nelem*ndim)

        model.probe_pos = nengo.Probe(model.arm_pos)  # track hand (x,y)

        adapt = nengo.Ensemble(n_neurons=1000, dimensions=ndim)  # , radius=np.sqrt(2))

        learn_conn = nengo.Connection(
            adapt, model.arm_node,
            function=lambda x: np.zeros(ndim),
            learning_rule_type=nengo.PES(learning_rate),
            synapse=0.05)


        # Calculate the error signal with another ensemble
        model.error = nengo.Ensemble(100, dimensions=ndim)

        # Error = actual - target = post - pre
        nengo.Connection(model.arm_node, model.error)
        # Here transform multiply target node with -1 and adds to model error
        nengo.Connection(target_node, model.error, transform=-1)

        model.probe_error = nengo.Probe(model.error)
        nengo.Connection(model.error, learn_conn.learning_rule)

        return model

SNN = build_model()
