import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv, MessagePassing
from torch_geometric.nn.norm import BatchNorm

from pd_mesh_net.nn.conv import PrimalConv, GATConvNoSelfLoops


class DualPrimalConv(torch.nn.Module):
    r"""Implements a modified version of the convolution operation defined on
    dual-primal graph convolutional neural networks from the
    `"Dual-Primal Graph Convolutional Networks"
    <https://arxiv.org/abs/1806.00770>`_ paper, which can work with the
    "medial graph"-"simplex mesh" pair defined on a triangular mesh.
    Uses the graph-attention networks defined in torch_geometric.nn.conv.GatConv
    as a basic unit.

    Args:
        in_channels_primal, in_channels_dual (int): Size of each input sample
            from the primal/dual graph respectively.
        out_channels_primal, out_channels_dual (int): Size of each output sample
            for the primal/dual graph respectively.
        single_dual_nodes (bool): If True, dual graphs are assumed to have
            single nodes; otherwise, they are assumed to have double nodes. Cf.
            :obj:`pd_mesh_net.utils.GraphCreator`.
        undirected_dual_edges (bool): If True, every directed edge in the dual
            graphs is assumed to have an opposite directed edge; otherwise,
            directed edges in the dual graphs are assumed not to have an
            opposite directed edge. Cf. :obj:`pd_mesh_net.utils.GraphCreator`.
        heads (int, optional): Number of attention heads associated to the
            primal and dual graphs. (default: :obj:`1`)
        concat_primal, concat_dual (bool, optional): If set to :obj:`False`, the
            attention heads associated respectively to the primal and to the
            dual graph are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope_primal, negative_slope_dual (float, optional): LeakyReLU
            angle of the negative slope, respectively for the layer associated
            to the primal graph and for the layer associated to the dual graph.
            (default: :obj:`0.2`)
        dropout_primal, dropout_dual (float, optional): Dropout probability of
            the normalized attention coefficients which exposes each node,
            respectively of the primal and of the dual graph, to a
            stochastically sampled neighborhood during training.
            (default: :obj:`0`)
        bias_primal, bias_dual (bool, optional): If set to :obj:`False`, the
            layer associated respectively to the primal and to the dual graph
            will not learn an additive bias. (default: :obj:`True`)
        add_self_loops_to_dual_graph (bool, optional): If set to :obj:`True`, a
            regular graph-attention convolutional layer is instantiated for the
            dual graph, thus self-loops are added to the dual graph. If set to
            :obj:`False` instead, a modified version of the graph-attention
            convolutional layer is instantiated for the dual graph, with no
            addition of self-loops to the latter. (default: :obj:`False`)
    
    Attributes:
        input_parameters (dict): Stores the value associated to each of the
            above input arguments when the instance is created.
        primal_attention_coefficients (torch.Tensor of shape
            :obj:`[num_primal_edges, num_attention_heads]`, where
            `num_primal_edges` is the number of edges in the primal graph
            and `num_attention_heads` is the number of attention heads): The
            i-th element stores the attention coefficient associated to the i-th
            edge in the edge-index matrix of the primal graph.
        _dual_layer (torch.geometric.nn.conv.GATConv or
            pd_mesh_net.nn.conv.GATConvNoSelfLoops): Dual-graph convolutional
            layer.
        _primal_layer (pd_mesh_net.nn.conv.PrimalConv): Primal-graph
            convolutional layer.
    """

    def __init__(self,
                 in_channels_primal,
                 in_channels_dual,
                 out_channels_primal,
                 out_channels_dual,
                 single_dual_nodes,
                 undirected_dual_edges,
                 heads=1,
                 concat_primal=True,
                 concat_dual=True,
                 negative_slope_primal=0.2,
                 negative_slope_dual=0.2,
                 dropout_primal=0,
                 dropout_dual=0,
                 bias_primal=True,
                 bias_dual=True,
                 add_self_loops_to_dual_graph=False):
        if (single_dual_nodes):
            assert (undirected_dual_edges), (
                "The dual-graph configuration with single dual nodes and "
                "directed dual edges is not valid. Please specify a different "
                "configuration.")
        super(DualPrimalConv, self).__init__()
        # Save input parameters as an attribute.
        self.__input_parameters = {
            k: v for k, v in locals().items() if (k[0] != '_' and k != 'self')
        }

        # The operation performed on the dual graph is a standard GAT
        # convolution.
        if (add_self_loops_to_dual_graph):
            # NOTE: Self-loops are added in the dual graph, and are used both
            # when computing attention coefficients and when performing feature
            # aggregation.
            self._dual_layer = GATConv(in_channels=in_channels_dual,
                                       out_channels=out_channels_dual,
                                       heads=heads,
                                       concat=concat_dual,
                                       negative_slope=negative_slope_dual,
                                       dropout=dropout_dual,
                                       bias=bias_dual)
        else:
            # NOTE: No self-loops are added in the dual graph.
            self._dual_layer = GATConvNoSelfLoops(
                in_channels=in_channels_dual,
                out_channels=out_channels_dual,
                heads=heads,
                concat=concat_dual,
                negative_slope=negative_slope_dual,
                dropout=dropout_dual,
                bias=bias_dual)

        # The operation performed on the primal graph is a modified version of
        # GAT convolution that uses dual features to compute attention
        # coefficients. NOTE: PrimalConv has a modified forward() method that
        # does not insert self-loops (as they would not find a correspondence in
        # the dual graph).
        self._primal_layer = PrimalConv(
            in_channels=in_channels_primal,
            out_channels=out_channels_primal,
            out_channels_dual=out_channels_dual,
            heads=heads,
            concat=concat_primal,
            concat_dual=concat_dual,
            negative_slope=negative_slope_primal,
            dropout=dropout_primal,
            bias=bias_primal,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges)

    @property
    def input_parameters(self):
        return self.__input_parameters

    @property
    def primal_attention_coefficients(self):
        assert (self.__primal_attention_coefficients is not None), (
            "Primal attention coefficients have not been computed yet. Please "
            "run convolution (forward()) first.")
        return self.__primal_attention_coefficients

    def forward(self, x_primal, x_dual, edge_index_primal, edge_index_dual,
                primal_edge_to_dual_node_idx):
        r"""Performs the convolution operation on the dual-primal network, by
        first performing a GATConv convolution on the dual graph and then
        carrying out a modified GATConv convolution on the primal graph, in
        which the attention coefficients are computed based on the node features
        of the dual graph.

        Args:
            x_primal (torch.Tensor of shape
                :obj:`[num_primal_nodes, in_channels]`, where `num_primal_nodes`
                is the number of input nodes of the primal graph and
                `in_channels` is the dimensionality of the input node features
                of the primal graph): Input node features of the primal graph.
            x_dual (torch.Tensor of shape
                :obj:`[num_dual_nodes, in_channels_dual]`, where
                `num_dual_nodes` is the number of input nodes of the associated
                dual graph and `in_channels_dual` is the dimensionality of the
                input node features of the associated dual graph): Input node
                features of the dual graph. The output of the convolution on
                these features in the dual layer will be used to compute the
                attention coefficients of the primal graph.
            edge_index_primal (torch.Tensor of shape :obj:`[2, num_edges]`]):
                List of the edges of the primal graph.
            edge_index_dual (torch.Tensor of shape :obj:`[2, num_edges]`]):
                List of the edges of the dual graph.
            primal_edge_to_dual_node_idx (dict): Dictionary that associates a
                tuple, encoding an edge e in the primal graph, to the index of
                the node in the dual graph that corresponds to the edge e.

        Returns:
            x_primal (torch.Tensor of shape
                :obj:`[num_primal_nodes, out_channels_primal]`, where
                `num_primal_nodes` is the number of nodes of the primal graph
                and `out_channels_primal` is the dimensionality of the output
                node features of the primal graph): Output node features of the
                primal graph.
            x_dual (torch.Tensor of shape
                :obj:`[num_dual_nodes, out_channels_dual]`, where
                `num_dual_nodes` is the number of nodes of the dual graph and
                `out_channels_dual` is the dimensionality of the output node
                features of the dual graph): Output node features of the dual
                graph.
        """
        # Convolution on the dual graph.
        x_dual = F.relu(self._dual_layer(x_dual, edge_index_dual))
        # Convolution on the primal graph.
        (x_primal_before_relu,
         primal_attention_coefficients) = self._primal_layer(
             x_primal, x_dual, edge_index_primal, primal_edge_to_dual_node_idx)
        x_primal = F.relu(x_primal_before_relu)

        self.__primal_attention_coefficients = primal_attention_coefficients

        return x_primal, x_dual


class DualPrimalResConv(torch.nn.Module):
    r"""Dual-primal mesh-convolution layer with skip connection. A first
    DualPrimalConv layer is followed by a chain of batch-normalization and
    DualPrimalConv layers; the output of the first DualPrimalConv layer is added
    to the output of the chain. Each batch-normalization layer, as well as the
    final output of the layer, has ReLU activation function. Based on the
    structure of `MResConv` from MeshCNN
    (https://github.com/ranahanocka/MeshCNN/).

    Args:
        in_channels_primal, in_channels_dual (int): Number of input channels in
            the primal/dual convolutional layer respectively.
        out_channels_primal, out_channels_dual (int): Number of output channels
            in the primal/dual convolutional layer respectively.
        heads (int): Number of attention heads associated to each of the primal
            and dual convolutional layers.
        concat_primal, concat_dual (bool): If set to :obj:`False`, the attention
            heads associated respectively to the primal and to the dual
            convolutional layers are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope_primal, negative_slope_dual (float): LeakyReLU angle of
            the negative slope, respectively for the layers associated to the
            primal graph and for the layers associated to the dual graph.
            (default: :obj:`0.2`)
        dropout_primal, dropout_dual (float): Dropout probability of the
            normalized attention coefficients which exposes each node,
            respectively of the primal and of the dual convolutional layers, to
            a stochastically sampled neighborhood during training.
            (default: :obj:`0`)
        bias_primal, bias_dual (bool): If set to :obj:`False`, the layers
            associated respectively to the primal and to the dual graph will not
            learn an additive bias. (default: :obj:`False`)
        single_dual_nodes (bool): If True, dual graphs are assumed to have
            single nodes; otherwise, they are assumed to have double nodes. Cf.
            :obj:`pd_mesh_net.utils.GraphCreator`.
        undirected_dual_edges (bool): If True, every directed edge in the dual
            graphs is assumed to have an opposite directed edge; otherwise,
            directed edges in the dual graphs are assumed not to have an
            opposite directed edge. Cf. :obj:`pd_mesh_net.utils.GraphCreator`.
        add_self_loops_to_dual_graph (bool): If set to :obj:`True`, regular
            graph-attention convolutional layers are instantiated as dual
            convolutional layers, thus self-loops are added to the dual graph.
            If set to :obj:`False` instead, a modified version of the
            graph-attention convolutional layer is instantiated for the dual
            graph, with no addition of self-loops to the latter.
        num_skips (int, optional): Number of consecutive batch-normalization and
            DualPrimalConv layers before the skip connection.
            (default: :obj:`1`)

    Attributes:
        conv0 (pd_mesh_net.nn.conv.DualPrimalConv): First dual-primal
            mesh-convolution layer.
        primal_attention_coefficients (torch.Tensor of shape
            :obj:`[num_primal_edges, num_attention_heads]`, where
            `num_primal_edges` is the number of edges in the primal graph
            and `num_attention_heads` is the number of attention heads): The
            i-th element stores the attention coefficient associated to the i-th
            edge in the edge-index matrix of the primal graph in the last
            DualPrimalConv layer.
        ---
        bn{i}_primal/dual (torch_geometric.nn.BatchNorm), `i` in
            `{1, ..., self.num_skips}`: `i`-th batch-normalization layer,
            respectively for the primal and for the dual features. Follows
            `self.conv{i - 1}` and has `self.out_channels_primal`/
            `self.out_channels_dual` output channels.
        conv{i} (pd_mesh_net.nn.conv.DualPrimalConv), `i` in
            `{1, ..., self.num_skips}`: `i`-th dual-primal mesh-convolution
            layer. Follows `self.bn_primal/dual{i - 1}` and has
            `self.out_channels_primal` and `self.out_channels_dual` output
            channels for the primal and for the dual features respectively.
    """

    def __init__(self,
                 in_channels_primal,
                 in_channels_dual,
                 out_channels_primal,
                 out_channels_dual,
                 heads,
                 concat_primal,
                 concat_dual,
                 negative_slope_primal,
                 negative_slope_dual,
                 dropout_primal,
                 dropout_dual,
                 bias_primal,
                 bias_dual,
                 single_dual_nodes,
                 undirected_dual_edges,
                 add_self_loops_to_dual_graph,
                 num_skips=1):
        super(DualPrimalResConv, self).__init__()

        self.__num_skips = num_skips
        # First dual-primal mesh-convolutional layer.
        self.conv0 = DualPrimalConv(
            in_channels_primal=in_channels_primal,
            in_channels_dual=in_channels_dual,
            out_channels_primal=out_channels_primal,
            out_channels_dual=out_channels_dual,
            heads=heads,
            concat_primal=concat_primal,
            concat_dual=concat_dual,
            negative_slope_primal=negative_slope_primal,
            negative_slope_dual=negative_slope_dual,
            dropout_primal=dropout_primal,
            dropout_dual=dropout_dual,
            bias_primal=bias_primal,
            bias_dual=bias_dual,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            add_self_loops_to_dual_graph=add_self_loops_to_dual_graph)

        for skip_idx in range(self.__num_skips):
            # Batch normalization layer.
            out_channels_primal_prev_layer = out_channels_primal
            if (concat_primal):
                out_channels_primal_prev_layer *= heads
            out_channels_dual_prev_layer = out_channels_dual
            if (concat_dual):
                out_channels_dual_prev_layer *= heads

            setattr(self, f'bn{skip_idx+1}_primal',
                    BatchNorm(out_channels_primal_prev_layer))
            setattr(self, f'bn{skip_idx+1}_dual',
                    BatchNorm(out_channels_dual_prev_layer))
            # Dual-primal mesh-convolutional layer.
            setattr(
                self, f'conv{skip_idx+1}',
                DualPrimalConv(
                    in_channels_primal=out_channels_primal_prev_layer,
                    in_channels_dual=out_channels_dual_prev_layer,
                    out_channels_primal=out_channels_primal,
                    out_channels_dual=out_channels_dual,
                    heads=heads,
                    concat_primal=concat_primal,
                    concat_dual=concat_dual,
                    negative_slope_primal=negative_slope_primal,
                    negative_slope_dual=negative_slope_dual,
                    dropout_primal=dropout_primal,
                    dropout_dual=dropout_dual,
                    bias_primal=bias_primal,
                    bias_dual=bias_dual,
                    single_dual_nodes=single_dual_nodes,
                    undirected_dual_edges=undirected_dual_edges,
                    add_self_loops_to_dual_graph=add_self_loops_to_dual_graph))

    @property
    def primal_attention_coefficients(self):
        return getattr(self,
                       f'conv{self.__num_skips}').primal_attention_coefficients

    def forward(self, x_primal, x_dual, edge_index_primal, edge_index_dual,
                primal_edge_to_dual_node_idx):
        r"""Forward pass, implements the actual skip connections.

        Args:
            x_primal (torch.Tensor of shape
                :obj:`[num_primal_nodes, in_channels]`, where `num_primal_nodes`
                is the number of input nodes of the primal graph and
                `in_channels` is the dimensionality of the input node features
                of the primal graph): Input node features of the primal graph.
            x_dual (torch.Tensor of shape
                :obj:`[num_dual_nodes, in_channels_dual]`, where
                `num_dual_nodes` is the number of input nodes of the associated
                dual graph and `in_channels_dual` is the dimensionality of the
                input node features of the associated dual graph): Input node
                features of the dual graph. The output of the convolution on
                these features in the dual layer will be used to compute the
                attention coefficients of the primal graph.
            edge_index_primal (torch.Tensor of shape :obj:`[2, num_edges]`]):
                List of the edges of the primal graph.
            edge_index_dual (torch.Tensor of shape :obj:`[2, num_edges]`]):
                List of the edges of the dual graph.
            primal_edge_to_dual_node_idx (dict): Dictionary that associates a
                tuple, encoding an edge e in the primal graph, to the index of
                the node in the dual graph that corresponds to the edge e.

        Returns:
            x_primal (torch.Tensor of shape
                :obj:`[num_primal_nodes, out_channels_primal]`, where
                `num_primal_nodes` is the number of nodes of the primal graph
                and `out_channels_primal` is the dimensionality of the output
                node features of the primal graph): Output node features of the
                primal graph.
            x_dual (torch.Tensor of shape
                :obj:`[num_dual_nodes, out_channels_dual]`, where
                `num_dual_nodes` is the number of nodes of the dual graph and
                `out_channels_dual` is the dimensionality of the output node
                features of the dual graph): Output node features of the dual
                graph.
        """
        # Apply the first convolution.
        x_primal, x_dual = self.conv0(
            x_primal=x_primal,
            x_dual=x_dual,
            edge_index_primal=edge_index_primal,
            edge_index_dual=edge_index_dual,
            primal_edge_to_dual_node_idx=primal_edge_to_dual_node_idx)
        x1_primal = x_primal
        x1_dual = x_dual
        # Apply the chain of batch-normalization (followed by ReLU) and
        # dual-primal mesh-convolution layers to the output of the first
        # convolution.
        for skip_idx in range(self.__num_skips):
            x_primal = getattr(self, f'bn{skip_idx+1}_primal')(F.relu(x_primal))
            x_dual = getattr(self, f'bn{skip_idx+1}_dual')(F.relu(x_dual))
            x_primal, x_dual = getattr(self, f'conv{skip_idx+1}')(
                x_primal=x_primal,
                x_dual=x_dual,
                edge_index_primal=edge_index_primal,
                edge_index_dual=edge_index_dual,
                primal_edge_to_dual_node_idx=primal_edge_to_dual_node_idx)
        if (self.__num_skips > 0):
            # Add the output of the first convolution to the output of the
            # chain, and apply ReLU.
            x_primal += x1_primal
            x_dual += x1_dual

        x_primal = F.relu(x_primal)
        x_dual = F.relu(x_dual)

        return x_primal, x_dual