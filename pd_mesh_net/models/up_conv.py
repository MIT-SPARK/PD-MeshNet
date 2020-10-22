import torch
import torch.nn.functional as F
from torch_geometric.nn.norm import BatchNorm

from pd_mesh_net.nn.conv import DualPrimalConv
from pd_mesh_net.nn.unpool import DualPrimalEdgeUnpooling


class DualPrimalUpConv(torch.nn.Module):
    r"""Performs an up-convolution operation. A dual-primal graph convolution is
    performed on the input graphs, bringing their features to a requested number
    of channels. Then, the graph connectivity is reconstructed by reverting a
    pooling operation, the parameters of which are given as input. The node
    features are assigned based on the graphs inputted to the unpooling layer
    (cf. `pd_mesh_net.nn.unpool.DualPrimalEdgeUnpooling` for further details).
    A further dual-primal graph convolution is applied to the reconstructed
    graphs. Optionally, the features from the original graph (before pooling)
    can be concatenated to the features of the reconstructed graph before
    inputting them to the last dual-primal convolutional layer. This can be
    useful when inserting the layer in a UNet-like architecture. The last
    dual-primal convolutional layer does not change the number of feature
    channels (without considering the optionally-concatenated ones). Both
    dual-primal convolutional layers are followed by a batch-normalization
    layer and ReLU activation.
        
    Partially based on an analogous model from MeshCNN
    (https://github.com/ranahanocka/MeshCNN/).

    Args:
        in_channels_primal/dual (int): Number of input channels of the primal
            and dual layers respectively (i.e., number of channels of the graphs
            outputted by the previous layer).
        out_channels_primal/dual (int): Number of output channels of the primal
            and dual layers respectively (i.e., number of output channels that
            the primal/dual graph had before the pooling operation that should
            be reverted was performed).
        single_dual_nodes (bool): If True, dual graphs are assumed to have
            single nodes; otherwise, they are assumed to have double nodes. Cf.
            :obj:`pd_mesh_net.utils.GraphCreator`.
        undirected_dual_edges (bool): If True, every directed edge in the dual
            graphs is assumed to have an opposite directed edge; otherwise,
            directed edges in the dual graphs are assumed not to have an
            opposite directed edge. Cf. :obj:`pd_mesh_net.utils.GraphCreator`.
        attention_heads_convolution (int, optional): Number of attention heads
            to use in the convolution layer. (default: :obj:`1`)
        concat_data_from_before_pooling (bool, optional): If True, the features
            from the original graph, before pooling, are concatenated to the
            features obtained with the unpooling layer. Cf. docs above.
            (default: :obj:`True`)
    
    Attributes:
        None.
    """

    def __init__(self,
                 in_channels_primal,
                 in_channels_dual,
                 out_channels_primal,
                 out_channels_dual,
                 single_dual_nodes,
                 undirected_dual_edges,
                 attention_heads_convolution=1,
                 concat_data_from_before_pooling=True):
        assert (attention_heads_convolution == 1
               ), "Only single-head attention is currently supported."
        self.__concat_data_from_before_pooling = concat_data_from_before_pooling
        super(DualPrimalUpConv, self).__init__()
        # Add the first dual-primal convolutional layer, that restores the
        # number of channels of the original graph.
        self.__conv1 = DualPrimalConv(
            in_channels_primal=in_channels_primal,
            in_channels_dual=in_channels_dual,
            out_channels_primal=out_channels_primal,
            out_channels_dual=out_channels_dual,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            heads=attention_heads_convolution,
            concat_primal=False,
            concat_dual=False)
        # Add the unpooling layer.
        self.__unpool = DualPrimalEdgeUnpooling(
            out_channels_dual=out_channels_dual)
        # Add the second dual-primal convolutional layer.
        in_channels_conv2_primal = out_channels_primal
        in_channels_conv2_dual = out_channels_dual
        if (concat_data_from_before_pooling):
            in_channels_conv2_primal *= 2
            in_channels_conv2_dual *= 2
        self.__conv2 = DualPrimalConv(
            in_channels_primal=in_channels_conv2_primal,
            in_channels_dual=in_channels_conv2_dual,
            out_channels_primal=out_channels_primal,
            out_channels_dual=out_channels_dual,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            heads=attention_heads_convolution,
            concat_primal=False,
            concat_dual=False)
        # Add the batch-normalization layers.
        self.__bn_primal1 = BatchNorm(in_channels=out_channels_primal)
        self.__bn_primal2 = BatchNorm(in_channels=out_channels_primal)
        self.__bn_dual1 = BatchNorm(in_channels=out_channels_dual)
        self.__bn_dual2 = BatchNorm(in_channels=out_channels_dual)

    def forward(self,
                primal_graph_batch,
                dual_graph_batch,
                primal_edge_to_dual_node_idx_batch,
                pooling_log,
                primal_graph_batch_before_pooling=None,
                dual_graph_batch_before_pooling=None):
        r"""Performs the up-convolution operation (cf. class docs for further
        details).

        Args:
            primal_graph_batch (torch_geometric.data.batch.Batch): Batch
                containing the input primal graphs on which the up-convolution
                operation should be applied.
            dual_graph_batch (torch_geometric.data.batch.Batch): Batch
                containing the input dual graphs on which the up-convolution
                operation should be applied.
            primal_edge_to_dual_node_idx_batch (dict): Dictionary representing
                the associations between primal-graph edges and dual-graph nodes
                in the input batch.
            pooling_log (pd_mesh_net.nn.pool.PoolingInfo): Data structure
                containing the information saved when pooling, and needed to
                perform the unpooling operation.
            primal_graph_batch_before_pooling (torch_geometric.data.batch.Batch,
                optional): Only required if the class argument
                `concat_data_from_before_pooling` is True, in which case it is
                the batch containing the original primal graphs before pooling.
                (default: :obj:`None`)
            dual_graph_batch_before_pooling (torch_geometric.data.batch.Batch,
                optional): Only required if the class argument
                `concat_data_from_before_pooling` is True, in which case it is
                the batch containing the original dual graphs before pooling.
                (default: :obj:`None`)

        Returns:
            new_primal_graph_batch (torch_geometric.data.batch.Batch): Output
                primal-graph batch after the up-convolution operation.
            new_dual_graph_batch (torch_geometric.data.batch.Batch): Output
                dual-graph batch after the up-convolution operation.
            new_primal_edge_to_dual_node_idx_batch (dict): Output dictionary
                representing the associations between primal-graph edges and
                dual-graph nodes in the batch after unpooling.
        """
        if (self.__concat_data_from_before_pooling):
            assert (
                primal_graph_batch_before_pooling is not None and
                dual_graph_batch_before_pooling is not None
            ), ("When the class argument `concat_data_from_before_pooling` is "
                "True, it is required to also input the primal and dual graph "
                "as they were before performing the pooling operation.")
        # Perform the first convolution operation, followed by
        # batch-normalization and ReLU activation.
        primal_graph_batch.x, dual_graph_batch.x = self.__conv1(
            x_primal=primal_graph_batch.x,
            x_dual=dual_graph_batch.x,
            edge_index_primal=primal_graph_batch.edge_index,
            edge_index_dual=dual_graph_batch.edge_index,
            primal_edge_to_dual_node_idx=primal_edge_to_dual_node_idx_batch)
        primal_graph_batch.x = F.relu(self.__bn_primal1(primal_graph_batch.x))
        dual_graph_batch.x = F.relu(self.__bn_dual1(dual_graph_batch.x))
        # Perform the unpooling operation.
        (new_primal_graph_batch, new_dual_graph_batch,
         new_primal_edge_to_dual_node_idx_batch) = self.__unpool(
             primal_graph_batch=primal_graph_batch,
             dual_graph_batch=dual_graph_batch,
             pooling_log=pooling_log)

        # If required, concatenate the features from the original graph.
        if (self.__concat_data_from_before_pooling):
            assert (new_primal_graph_batch.x.shape ==
                    primal_graph_batch_before_pooling.x.shape)
            assert (new_dual_graph_batch.x.shape ==
                    dual_graph_batch_before_pooling.x.shape)
            new_primal_graph_batch.x = torch.cat(
                [new_primal_graph_batch.x, primal_graph_batch_before_pooling.x],
                dim=1)
            new_dual_graph_batch.x = torch.cat(
                [new_dual_graph_batch.x, dual_graph_batch_before_pooling.x],
                dim=1)

        # Perform the second convolution operation, followed by
        # batch-normalization and ReLU activation.
        new_primal_graph_batch.x, new_dual_graph_batch.x = self.__conv2(
            x_primal=new_primal_graph_batch.x,
            x_dual=new_dual_graph_batch.x,
            edge_index_primal=new_primal_graph_batch.edge_index,
            edge_index_dual=new_dual_graph_batch.edge_index,
            primal_edge_to_dual_node_idx=new_primal_edge_to_dual_node_idx_batch)
        new_primal_graph_batch.x = F.relu(
            self.__bn_primal2(new_primal_graph_batch.x))
        new_dual_graph_batch.x = F.relu(self.__bn_dual2(new_dual_graph_batch.x))

        return (new_primal_graph_batch, new_dual_graph_batch,
                new_primal_edge_to_dual_node_idx_batch)
