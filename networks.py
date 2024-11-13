import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
import numpy as np
from network_utils import gumbel_softmax, onehot_from_logits


class ProtoLayer(nn.Module):
    def __init__(self, comm_dim, num_prototypes):
        super(ProtoLayer, self).__init__()
        self.prototypes = nn.Parameter(data=torch.Tensor(num_prototypes, comm_dim))
        # Initialize by manually setting such that prototypes are close to I matrix.
        nn.init.uniform_(self.prototypes, -2, 2)
        # nn.init.zeros_(self.prototypes)
        #
        # # some random initialisation has been given to the prototypes.
        # for i in range(num_prototypes):
        #     for j in range(comm_dim):
        #         if i % comm_dim == j:
        #             self.prototypes.data[i, j] = 5
        #         else:
        #             self.prototypes.data[i, j] = -5
        #         self.prototypes.data[i, j] += np.random.random()

    def forward(self, comm_vectors):
        """ Calculates L2 distances from each latent comm_vector to each prototype. Useful for computing loss.
        Note:
            Uses fancy matrix computation technique that is not immediately obvious in the paper
            https://github.com/OscarcarLi/PrototypeDL/blob/master/autoencoder_helpers.py
        """
        constrained_protos = torch.sigmoid(self.prototypes)
        comms_squared = ProtoLayer.get_norms(comm_vectors).view(-1, 1)
        protos_squared = ProtoLayer.get_norms(constrained_protos).view(1, -1)
        dists_to_protos = comms_squared + protos_squared - 2 * torch.matmul(comm_vectors,
                                                                            torch.transpose(constrained_protos, 0, 1))

        alt_comms_squared = ProtoLayer.get_norms(comm_vectors).view(1, -1)
        alt_protos_squared = ProtoLayer.get_norms(constrained_protos).view(-1, 1)
        dists_to_comms = alt_comms_squared + alt_protos_squared - 2 * torch.matmul(constrained_protos,
                                                                                   torch.transpose(comm_vectors,
                                                                                                   0, 1))
        return [dists_to_protos, dists_to_comms]

    def get_normalized_prototypes(self):
        return torch.sigmoid(self.prototypes)

    @staticmethod
    def get_norms(x):
        return torch.sum(torch.pow(x, 2), dim=1)


class ProtoNetwork(nn.Module):
    # Static variable used to get information before an instance is created. Kind of gross
    # num_protos = None

    def __init__(self, input_dim, out_dim, discrete, num_layers=2, hidden_dim=64, num_protos=6, constrain_out=False):
        super(ProtoNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            out = hidden_dim
            if layer_idx == num_layers - 1:
                # out = ProtoNetwork.num_protos if discrete else out_dim
                out = num_protos if discrete else out_dim
            self.layers.append(nn.Linear(in_dim, out))
        self.discrete = discrete
        self.dropout = nn.Dropout(0.3)
        # self.prototype_layer = ProtoLayer(out_dim, num_prototypes=ProtoNetwork.num_protos)  # FIXME. Shouldn't be static
        self.prototype_layer = ProtoLayer(out_dim, num_prototypes=num_protos)
        if constrain_out and not discrete:
            # initialize small to prevent saturation
            self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = torch.tanh
        else:
            self.out_fn = lambda x: x

    # There's something a bit awkward here. For a discrete proto network, the forward() call outputs the logits for
    # distribution over prototypes, not the prototypes themselves. That's fine, but it means that the decoded stuff
    # via communication isn't actually taking in the communication itself. One could fix this by manually inserting
    # the prototype multiplication in there if desired, but that feels odd.
    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        internal = X
        for layer_id, layer in enumerate(self.layers):
            internal = layer(internal)
            if layer_id != len(self.layers) - 1:
                internal = F.relu(internal)
        output = self.out_fn(internal)
        return output

    def step(self, raw_output, explore, exploration, device):
        if self.discrete:
            # Having higher temperature seems to help encourage using more prototypes. But I'm not sure yet what's best.
            masked = self.dropout(raw_output)
            onehot_pred = gumbel_softmax(masked, temperature=1, hard=True) if explore else onehot_from_logits(raw_output)
            # Multiply onehot by prototypes
            # print(onehot_pred)
            prototypes = torch.sigmoid(self.prototype_layer.prototypes)
            multiplied = torch.matmul(onehot_pred, prototypes)
            return multiplied
        if explore:
            return raw_output + Variable(Tensor(exploration.noise()), requires_grad=False).to(device)
        return self.__convert_to_prototype__(raw_output)

    def onehot_step(self, raw_output, explore):
        assert self.discrete
        masked = self.dropout(raw_output)
        onehot_pred = gumbel_softmax(masked, temperature=1, hard=True) if explore else onehot_from_logits(raw_output)
        return onehot_pred

    def __convert_to_prototype__(self, output):
        dists_to_protos, _ = self.prototype_layer(output)
        closest_proto_idx = torch.argmin(dists_to_protos, dim=1)
        closest_proto = torch.sigmoid(self.prototype_layer.prototypes[closest_proto_idx])
        return closest_proto
