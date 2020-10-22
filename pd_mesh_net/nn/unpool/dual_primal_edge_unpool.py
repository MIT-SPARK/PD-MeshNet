import torch
from torch_geometric.data import Batch
from torch_geometric.nn.inits import glorot


class DualPrimalEdgeUnpooling(torch.nn.Module):
    r"""Unpools a `DualPrimalEdgePooling` operation, so as to restore the
    original connectivity. The data structures stored when performing the
    pooling operation are required as input. The primal and dual nodes that were
    merged upon pooling a primal edge and reconstructing the dual graph are
    assigned the same feature as the primal/dual node formed after pooling.
    The approach is somehow similar to the one used in
    `torch_geometric.nn.pool.EdgePooling`.

    Args:
        out_channels_dual (int): Number of output dual channels. Required to
            initialize the parameter that is assigned as feature to the dual
            nodes that did not have a corresponding dual node after pooling (cf.
            attribute `new_dual_feature`).

    Attributes:
        new_dual_feature (torch.Tensor of shape :obj:`[1, out_channels_dual]`,
            cf. argument `out_channels_dual`): Learnable parameter assigned as
            feature to the dual nodes recreated after unpooling that did not
            have a corresponding dual node after pooling.
    """

    def __init__(self, out_channels_dual):
        super(DualPrimalEdgeUnpooling, self).__init__()
        self.__out_channels_dual = out_channels_dual
        self.new_dual_feature = torch.nn.Parameter(
            torch.Tensor(1, out_channels_dual))
        # Initialize parameter.
        glorot(self.new_dual_feature)

    def forward(self, primal_graph_batch, dual_graph_batch, pooling_log):
        r"""Performs the unpooling operation.

        Args:
            primal_graph_batch (torch_geometric.data.batch.Batch): Batch
                containing the input primal graphs on which the unpooling
                operation should be applied.
            dual_graph_batch (torch_geometric.data.batch.Batch): Batch
                containing the input dual graphs on which the unpooling
                operation should be applied.
            pooling_log (pd_mesh_net.nn.pool.PoolingInfo): Data structure
                containing the information saved when pooling, and needed to
                perform the unpooling operation.

        Returns:
            new_primal_graph_batch (torch_geometric.data.batch.Batch): Output
                primal-graph batch after unpooling.
            new_dual_graph_batch (torch_geometric.data.batch.Batch): Output
                dual-graph batch after unpooling.
            new_primal_edge_to_dual_node_idx_batch (dict): Output dictionary
                representing the associations between primal-graph edges and
                dual-graph nodes in the batch after unpooling.
        """
        # Reconstruct the primal graph.
        old_primal_node_to_new_one = pooling_log.old_primal_node_to_new_one
        # - Reconstruct the connectivity.
        new_primal_graph_batch = Batch(batch=pooling_log.old_primal_graph_batch)
        new_primal_graph_batch.edge_index = pooling_log.old_primal_edge_index
        # - Assign to the new primal nodes the features of the primal nodes in
        #   which these were merged when performing the pooling operation.
        new_primal_graph_batch.x = primal_graph_batch.x[
            old_primal_node_to_new_one]

        # Reconstruct the dual graph.
        old_dual_node_to_new_one = pooling_log.old_dual_node_to_new_one
        assert (old_dual_node_to_new_one is not None), (
            "The input pooling log does not contain the mapping from dual "
            "nodes before pooling to dual nodes after pooling. Please set the "
            "argument `return_old_dual_node_to_new_dual_node` to True in the "
            "`DualPrimalEdgePooling` layer.")
        # - Reconstruct the connectivity.
        new_dual_graph_batch = Batch(batch=pooling_log.old_dual_graph_batch)
        new_dual_graph_batch.edge_index = pooling_log.old_dual_edge_index
        # - Assign the same learnable feature to all the dual nodes that do not
        #   have a corresponding dual node after performing the pooling
        #   operation.
        num_old_dual_nodes = len(old_dual_node_to_new_one)
        new_dual_graph_batch_x = self.new_dual_feature.repeat(
            num_old_dual_nodes, 1)
        # - Assign to the new dual nodes that have a corresponding dual node
        #   after the pooling operation the features of these corresponding
        #   nodes.
        new_dual_graph_batch_x[
            old_dual_node_to_new_one != -1] = dual_graph_batch.x[
                old_dual_node_to_new_one[old_dual_node_to_new_one != -1]]
        new_dual_graph_batch.x = new_dual_graph_batch_x

        # Remap the primal-graph edges to the dual-graph nodes.
        new_primal_edge_to_dual_node_idx_batch = (
            pooling_log.old_primal_edge_to_dual_node_index)

        return (new_primal_graph_batch, new_dual_graph_batch,
                new_primal_edge_to_dual_node_idx_batch)
