import numpy as np

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax


class PrimalConv(GATConv):
    r"""Implements a modified version of the graph-attention convolution defined
    in torch_geometric.nn.conv.GATConv, allowing to compute attention
    coefficients on a certain graph - referred to as primal graph - from the
    node features of an external dual graph.

    Args:
        in_channels (int): Size of each input sample from the primal graph.
        out_channels (int): Size of each output sample for the primal graph.
        out_channels_dual (int): Size of each output sample for the dual graph.
        concat_dual (bool): If set to :obj:`False`, it will be assumed that the
            attention heads of the dual graph have been averaged instead of
            being concatenated.
        single_dual_nodes (bool): If True, it is assumed that the associated
            dual graph has single nodes; otherwise, it is assumed that it has
            double nodes. Cf. :obj:`pd_mesh_net.utils.GraphCreator`.
        undirected_dual_edges (bool): If True, every directed edge in the
            associated dual graph is assumed to have an opposite directed edge;
            otherwise, directed edges in the associated dual graph are assumed
            not to have an opposite directed edge. Cf.
            :obj:`pd_mesh_net.utils.GraphCreator`.
        concat (bool, optional): If set to :obj:`False`, the attention heads are
            averaged instead of concatenated. (default: :obj:`True`)
        heads (int, optional): Number of attention heads. (default: :obj:`1`)
        negative_slope (float, optional): LeakyReLU angle of the negative slope.
            (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)       
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 out_channels_dual,
                 concat_dual,
                 single_dual_nodes,
                 undirected_dual_edges,
                 concat=True,
                 heads=1,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 **kwargs):
        if (not concat_dual and heads != 1):
            raise ValueError(
                "It is not possible to use multiple heads on the primal graph "
                "when dual attention heads have been averaged, because in this "
                "case the dual features can be put in correspondence only with "
                "a single head of the primal graph.")
        if (single_dual_nodes):
            assert (undirected_dual_edges), (
                "The dual-graph configuration with single dual nodes and "
                "directed dual edges is not valid. Please specify a different "
                "configuration.")
        # Initialization from GAT.
        assert (not "flow" in kwargs)
        super(PrimalConv, self).__init__(in_channels=in_channels,
                                         out_channels=out_channels,
                                         heads=heads,
                                         concat=concat,
                                         negative_slope=negative_slope,
                                         dropout=dropout,
                                         bias=bias,
                                         flow="source_to_target",
                                         **kwargs)

        self.__single_dual_nodes = single_dual_nodes
        self.__undirected_dual_edges = undirected_dual_edges
        self.__out_channels_dual = out_channels_dual
        # Initialize dual features, which will be passed in the forward method.
        self.__dual_features = None
        # The following attribute will store primal attention coefficients and
        # will be updated each time a forward pass of the network is called.
        self.__alpha = None

        # Override shape of attention parameter vector that will be used to
        # produce the attention coefficients. The latter will be computed as the
        # softmax of the leaky-ReLU of the dot product between this attention
        # parameter vector and the output features from the dual graph, over all
        # dual nodes associated to a certain primal node.
        self.att = Parameter(torch.Tensor(1, heads, out_channels_dual))

        # Note: all the parameters have already been reset in the constructor of
        # the parent class, but since attention parameters have been changed,
        # they need to be reinitialized.
        self.reset_attention_parameters()

    def forward(self,
                x_primal,
                x_dual,
                edge_index_primal,
                primal_edge_to_dual_node_idx,
                size=None):
        r"""Implements the convolution operation on the modified GAT, by using
        the node features of the dual graph to compute the attention
        coefficients.
    
        Args:
            x_primal (torch.Tensor of shape
                :obj:`[num_primal_nodes, in_channels]`, where `num_primal_nodes`
                is the number of input nodes of the primal graph and
                `in_channels` is the dimensionality of the input node features
                of the primal graph): Input node features of the primal graph.
            x_dual (torch.Tensor of shape
                :obj:`[num_dual_nodes, out_channels_dual]`, where
                `num_dual_nodes` is the number of output nodes of the associated
                dual graph and `out_channels_dual` is the dimensionality of the
                output node features of the associated dual graph): Output node
                features of the associated dual graph, that will be used to
                compute the attention coefficients.
            edge_index_primal (torch.Tensor of shape :obj:`[2, num_edges]`]):
                List of the edges of the primal graph.
            primal_edge_to_dual_node_idx (dict): Dictionary that associates a
                tuple, encoding an edge e in the primal graph, to the index of
                the node in the dual graph that corresponds to the edge e.
            size (int, optional): Number of nodes in the primal graph,
                optionally used to computed the 'scattered' softmax.
                (default: :obj:`None`)

        Returns:
            out_primal_features (torch.Tensor of shape
                :obj:`[num_primal_nodes, out_channels]`, where
                `num_primal_nodes` is the number of nodes of the primal graph
                and `out_channels` is the dimensionality of the output node
                features of the primal graph): Output node features of the
                primal graph.
            primal_attention_coefficients (torch.Tensor of shape
                :obj:`[num_primal_edges, num_attention_heads]`, where
                `num_primal_edges` is the number of edges in the primal graph
                and `num_attention_heads` is the number of attention heads): The
                i-th element stores the attention coefficient associated to the
                i-th edge in the edge-index matrix of the primal graph.
        """
        assert (isinstance(primal_edge_to_dual_node_idx, dict))
        # The part before the feature propagation is exactly the same as in the
        # parent class GATConv, since the primal features get multiplied by the
        # weight matrix. However, self-loops are not added, as the would not
        # have a corresponding node in the dual graph.
        if torch.is_tensor(x_primal):
            x_primal = torch.matmul(x_primal, self.weight)
        else:
            raise TypeError("Node features must be tensors.")
            x_primal = (None if x_primal[0] is None else torch.matmul(
                x_primal[0], self.weight), None if x_primal[1] is None else
                        torch.matmul(x_primal[1], self.weight))

        # Propagation is different, since the attention coefficients are here
        # computed based on the dual features.
        self.__dual_features = x_dual
        self.__primal_edge_to_dual_node_idx = primal_edge_to_dual_node_idx
        output_primal_features = self.propagate(edge_index=edge_index_primal,
                                                size=size,
                                                x=x_primal)

        primal_attention_coefficients = self.__alpha

        return output_primal_features, primal_attention_coefficients

    def reset_attention_parameters(self):
        glorot(self.att)

    def message(self, edge_index_i, edge_index_j, x_i, x_j, size_i):
        assert (x_i is not None)
        # Compute attention coefficients.
        assert (len(edge_index_i) == len(edge_index_j))
        edge_index_i_list = edge_index_i.tolist()
        edge_index_j_list = edge_index_j.tolist()
        if (self.__single_dual_nodes):
            # Dual-graph configuration A.
            # - Find index of the dual node {i, j} in the dual graph.
            indices_node_ji = [
                self.__primal_edge_to_dual_node_idx[tuple(
                    sorted([edge_i, edge_j]))]
                for edge_i, edge_j in zip(edge_index_i_list, edge_index_j_list)
            ]
        else:
            # Dual-graph configurations B and C.
            # - Find index of the dual node j->i in the dual graph.
            indices_node_ji = [
                self.__primal_edge_to_dual_node_idx[(edge_j, edge_i)]
                for edge_i, edge_j in zip(edge_index_i_list, edge_index_j_list)
            ]
        # Since when the dual attention heads are more than one they need to be
        # concatenated (cf. assert in constructor), we can here safely reshape
        # the dual features so as to separate the contributions from the
        # different heads, which will be as many as the primal heads.
        x_dual = self.__dual_features[indices_node_ji].view(
            -1, self.heads, self.__out_channels_dual)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        alpha = (x_dual * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Store attention coefficients.
        self.__alpha = alpha.view(-1, self.heads)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if (self.concat is True):
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if (self.bias is not None):
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
