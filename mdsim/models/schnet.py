"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import SchNet, radius_graph
from torch_scatter import scatter

from mdsim.common.registry import registry
from mdsim.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)

class SchNetDisc(SchNet):
    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,
        use_pbc=True,
        regress_forces=True,
        otf_graph=False,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        readout="add",
    ):
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph


        super(SchNetDisc, self).__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
        )

        self.lin1 = Linear(num_atoms*hidden_channels, 1)
        #self.lin2 = Linear(hidden_channels, 1)
        self.transition = Linear(hidden_channels+3, hidden_channels)
    
    def forward(self, z, pos, forces, batch=None):
            batch_size = 1+torch.max(batch).item()  
            assert z.dim() == 1 and z.dtype == torch.long
            batch = torch.zeros_like(z) if batch is None else batch

            h = self.embedding(z)
            #concatenate forces to the embeddings and bring down to same size again
            h = self.transition(torch.cat([h, forces], dim = -1).float())
            
        
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                    max_num_neighbors=self.max_num_neighbors)
            row, col = edge_index
            edge_weight = (pos[row] - pos[col]).norm(dim=-1)
            edge_attr = self.distance_expansion(edge_weight)

            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            #flatten each batch (flatten all atomic representations)
            h = h.reshape(batch_size, -1)
            
            #classification head
            h = self.lin1(h)
            # h = self.act(h)
            # h = self.lin2(h)
            h = F.sigmoid(h)
            
            return h


class SchNetPlusIntermediate(SchNet):
    """Modify SchNet to output intermediate atomic representations for variance prediction"""
        
    def forward(self, z, pos, batch=None):
            
            assert z.dim() == 1 and z.dtype == torch.long
            batch = torch.zeros_like(z) if batch is None else batch

            h = self.embedding(z)

            edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                    max_num_neighbors=self.max_num_neighbors)
            row, col = edge_index
            edge_weight = (pos[row] - pos[col]).norm(dim=-1)
            edge_attr = self.distance_expansion(edge_weight)

            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            rep = h
            h = self.lin1(h)
            h = self.act(h)
            h = self.lin2(h)

            if self.dipole:
                # Get center of mass.
                mass = self.atomic_mass[z].view(-1, 1)
                c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
                h = h * (pos - c.index_select(0, batch))

            if not self.dipole and self.mean is not None and self.std is not None:
                h = h * self.std + self.mean

            if not self.dipole and self.atomref is not None:
                h = h + self.atomref(z)

            out = scatter(h, batch, dim=0, reduce=self.readout)

            if self.dipole:
                out = torch.norm(out, dim=-1, keepdim=True)

            if self.scale is not None:
                out = self.scale * out

            return out, rep


@registry.register_model("schnet_policy")
class SchNetPolicy(SchNetPlusIntermediate):
    r"""Wrapper around the continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_. Each layer uses interaction
    block of the form:

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    Args:
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets (int): Number of targets to predict.
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`128`)
        num_filters (int, optional): Number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): Number of interaction blocks
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
    """

    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,
        use_pbc=True,
        regress_forces=True,
        otf_graph=False,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        readout="add",
    ):
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph

        super(SchNetPolicy, self).__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
        )

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch

        if self.otf_graph:
            edge_index, cell_offsets, _, neighbors = radius_graph_pbc(data, self.cutoff, 500)
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        if self.use_pbc:
            assert z.dim() == 1 and z.dtype == torch.long

            out = get_pbc_distances(
                data.pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.natoms,
            )

            edge_index = out["edge_index"]
            edge_weight = out["distances"]
            edge_attr = self.distance_expansion(edge_weight)

            h = self.embedding(z)
            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            rep = h
            #energy head
            h = self.lin1(h)
            h = self.act(h)
            h = self.lin2(h)

            batch = torch.zeros_like(z) if batch is None else batch
            energy = scatter(h, batch, dim=0, reduce=self.readout)
        else:
            energy, rep = super(SchNetPolicy, self).forward(z, pos, batch)
        return energy, rep

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)

        energy, rep = self._forward(data)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces, rep
        else:
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


@registry.register_model("schnet_discriminator")
class SchNetDiscriminator(SchNetDisc):
    r"""Wrapper around the continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_. Each layer uses interaction
    block of the form:

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    Args:
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets (int): Number of targets to predict.
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`128`)
        num_filters (int, optional): Number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): Number of interaction blocks
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
    """


    def forward(self, data):
        
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch
        forces = data.force
        batch_size = 1 + torch.max(batch).item()

        if self.otf_graph:
            edge_index, cell_offsets, _, neighbors = radius_graph_pbc(data, self.cutoff, 500)
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        if self.use_pbc:
            assert z.dim() == 1 and z.dtype == torch.long

            out = get_pbc_distances(
                data.pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.natoms,
            )

            edge_index = out["edge_index"]
            edge_weight = out["distances"]
            edge_attr = self.distance_expansion(edge_weight)

            h = self.embedding(z)
            #concatenate forces to the embeddings and bring down to same size again
            h = self.transition(torch.cat([h, forces], dim = -1).float())
            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            #flatten each batch (flatten all atomic representations)
            h = h.reshape(batch_size, -1)
            
            #classification head
            h = self.lin1(h)
            h = self.act(h)
            h = self.lin2(h)
            h = F.sigmoid(h)
            out = h

        
        else:
            out = super(SchNetDiscriminator, self).forward(z, pos, forces, batch)
        return out

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())



