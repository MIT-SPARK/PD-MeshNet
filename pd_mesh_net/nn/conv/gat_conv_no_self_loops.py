import torch
from torch_geometric.nn.conv import GATConv
from torch_geometric.utils import remove_self_loops


class GATConvNoSelfLoops(GATConv):
    r"""Implements a version of Graph-Attention convolution in which the only
    difference w.r.t. GATConv is that self loops are not added to the graph.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of attention heads. (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the attention heads are
            are averaged instead of concatenated. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
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
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 **kwargs):
        # Initialize the regular GATConv.
        super(GATConvNoSelfLoops, self).__init__(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 heads=heads,
                                                 concat=concat,
                                                 negative_slope=negative_slope,
                                                 dropout=dropout,
                                                 bias=bias,
                                                 **kwargs)

    def forward(self, x, edge_index, size=None):
        """"""
        # Code from GATConv. The only modification is that the part that caused
        # the addition of self-loops is omitted.
        edge_index, _ = remove_self_loops(edge_index)
        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)