from collections import namedtuple
import torch

from torch_scatter import scatter_add, scatter_mean
from torch_sparse import coalesce
from torch_geometric.data import Batch
from torch_geometric.utils import remove_self_loops

from pd_mesh_net.nn.pool import PoolingInfo
from pd_mesh_net.utils import NodeClustersWithUnionFind


class DualPrimalEdgePooling(torch.nn.Module):
    r"""Implements edge-pooling driven by the attention coefficients on the
    primal edges. Primal edges are pooled, causing the corresponding dual nodes
    to be removed from the dual graph.
    The approach is somehow similar to the one used in
    `torch_geometric.nn.pool.EdgePooling`.

    Args:
        self_loops_in_output_dual_graph (bool): If True, self loops are added in
            the dual graphs that will be returned after pooling; otherwise, the
            dual graphs returned after pooling will have no self-loops.
        single_dual_nodes (bool): If True, it will be expected that the input
            dual graphs have 'single' nodes (i.e., for all i < j, the primal
            edges i->j and j->i, both get mapped to the dual node {i, j}) and,
            likewise the output dual graphs will be created with single nodes;
            otherwise, it will be expected that the input dual graphs have
            'double' nodes, and the output dual graphs will be created with
            double nodes. Cf. :obj:`pd_mesh_net.utils.GraphCreator`.
        undirected_dual_edges (bool): If True, it will be assumed that every
            directed edge in the input dual graphs has an opposite directed
            edge, and same will apply for the output dual graphs; otherwise, it
            will be assumed that directed edges in the input dual graphs do not
            have an opposite directed edge, as same will apply for the output
            dual graphs. Cf. :obj:`pd_mesh_net.utils.GraphCreator`.
        num_primal_edges_to_keep (int, optional): If not None, the pooling layer
            will pool edges in the primal graph so that, for each sample in the
            path, this number of edges are left in the primal graph at the layer
            output; if for any sample this number is larger than the number of
            primal edges in the sample, no edges are pooled. The edges are
            pooled with priority given by the largest magnitude of the
            associated attention coefficient and by the lowest magnitude of the
            associated attention coefficient, respectively if
            `use_decreasing_attention_coefficient` is True or False. If None,
            one needs to specify the minimum/maximum attention coefficient
            associated to a primal edge for the latter to be collapsed (cf.
            argument `primal_att_coeff_threshold`) or the fraction of primal
            edges to keep (cf. argument `fraction_primal_edges_to_keep`). Note:
            in the case of faces that form a complete triangle fan in which only
            one edge between them is not pooled, also the latter edge is pooled.
            Therefore, the number of edges actually kept might be lower than
            `num_primal_edges_to_keep`. (default: :obj:`None`)
        fraction_primal_edges_to_keep (float, optional): If not None, the
            pooling layer will pool edges in the primal graph so that this
            fraction of edges are left in the primal graph at the layer output.
            The edges are pooled with priority given by the largest magnitude of
            the associated attention coefficient and by the lowest magnitude of
            the associated attention coefficient, respectively if
            `use_decreasing_attention_coefficient` is True or False. If None,
            one needs to specify the minimum/maximum attention coefficient
            associated to a primal edge for the latter to be collapsed (cf.
            argument `primal_att_coeff_threshold`) or the number of primal edges
            to keep (cf. argument `num_primal_edges_to_keep`). Note: in the case
            of faces that form a complete triangle fan in which only one edge
            between them is not pooled, also the latter edge is pooled.
            Therefore, the fraction of edges actually kept might be lower than
            `fraction_primal_edges_to_keep`. (default: :obj:`None`)
        primal_att_coeff_threshold (float, optional): If not None, the layer
            will pool all the primal edges with an associated attention
            coefficient larger or smaller than this value, respectively if the
            `use_decreasing_attention_coefficient` is True or False. If None,
            one needs to specify the amount of primal edges that the layer
            should leave in the primal graph (cf. arguments
            `num_primal_edges_to_keep` and `fraction_primal_edges_to_keep`).
            Note: in the case of faces that form a complete triangle fan in
            which only one edge between them is not pooled, also the latter edge
            is pooled. Therefore, also some edges that have an associated primal
            attention coefficient below/above `primal_att_coeff_threshold` might
            be pooled. (default: :obj:`None`)
        allow_pooling_consecutive_edges (bool, optional): If True, no
            restrictions are put on the primal edges that can be pooled. If
            False, a primal edge can only be pooled if no primal nodes to which
            it belongs have been pooled previously. Setting this argument to
            False is only compatible with top-K pooling (cf. arguments
            `num_primal_edges_to_keep` and `fraction_primal_edges_to_keep`).
            (default: :obj:`True`)
        add_to_primal_att_coeffs (float, optional): This is added to each
            primal-edge attention coefficient. A similar approach used in
            `torch_geometric.nn.pool.EdgePooling` 'greatly helped with unpool
            stability'. (default: :obj:`0.0`)
        aggr_primal_features_pooling (str, optional): Parameter of the optional
            pooling layer. If 'mean'/'add', the feature of each new primal node
            after pooling is obtained by respectively averaging, summing and
            multiplying the features of the primal nodes that get merged into
            that node. (default: :obj:`mean`)
        aggr_dual_features_pooling (str, optional): Parameter of the optional
            pooling layer. If 'mean'/'add', whenever a new dual node is obtained
            by aggregating multiple previous dual nodes its feature is obtained
            by respectively averaging and summing the features of the previous
            dual nodes. (default: :obj:`mean`)
        use_decreasing_attention_coefficient (bool, optional): When using
            pooling based on an amount of edges (cf. arguments
            `num_primal_edges_to_keep` and `fraction_primal_edges_to_keep`): if
            True, primal edges are pooled by decreasing magnitude of the
            associated attention coefficients; if False, primal edges are pooled
            by increasing magnitude of the associated attention coefficients.
            When using pooling based on attention-coefficient threshold (cf.
            argument `primal_att_coeff_threshold`): primal edges are pooled if
            the associated attention coefficients are above the threshold - if
            True - or below the threshold - if False. (default: :obj:`True`)
        return_old_dual_node_to_new_dual_node (bool, optional): If True, the
            element with key `old_dual_node_to_new_one` of the namedtuple
            outputted by the forward method is a tensor of length
            `new_input_dual_nodes`- where `num_input_dual_nodes` is the number
            of input dual nodes in the batch - the i-th element of which
            maps the i-th input dual node to its corresponding output dual node;
            if the i-th input dual node has no corresponding output dual node,
            it is mapped to -1. If False, the element with key
            `old_dual_node_to_new_one` of the namedtuple outputted by the
            forward method is is `None`. (default: :obj:`False`)
        log_ratio_new_old_primal_nodes (bool, optional): If True, the element
            with key `ratio_new_old_primal_nodes` of the namedtuple
            outputted by the forward method is a tensor of length `num_samples`
            - where `num_samples` is the number of samples in the batch - the
            i-th element of which contains the ratio between the number of
            primal nodes in the new, pooled graph and the number of primal nodes
            before pooling, for the i-th sample in the batch. Otherwise, it is
            `None`. (default: :obj:`False`)
        log_ratio_new_old_primal_edges (bool, optional): If True, the element
            with key `ratio_new_old_primal_edges` of the namedtuple
            outputted by the forward method is a tensor of length `num_samples`
            - where `num_samples` is the number of samples in the batch - the
            i-th element of which contains the ratio between the number of
            primal edges in the new, pooled graph and the number of primal edges
            before pooling, for the i-th sample in the batch. Otherwise, it is
            `None`. (default: :obj:`False`)

    Attributes:
        None.
    """

    def __init__(self,
                 self_loops_in_output_dual_graph,
                 single_dual_nodes,
                 undirected_dual_edges,
                 num_primal_edges_to_keep=None,
                 fraction_primal_edges_to_keep=None,
                 primal_att_coeff_threshold=None,
                 allow_pooling_consecutive_edges=True,
                 add_to_primal_att_coeffs=0.0,
                 aggr_primal_features='mean',
                 aggr_dual_features='mean',
                 use_decreasing_attention_coefficient=True,
                 return_old_dual_node_to_new_dual_node=False,
                 log_ratio_new_old_primal_nodes=False,
                 log_ratio_new_old_primal_edges=False):
        assert (len([
            arg for arg in [
                num_primal_edges_to_keep, fraction_primal_edges_to_keep,
                primal_att_coeff_threshold
            ] if arg is not None
        ]) == 1), (
            "Exactly one of the arguments `num_primal_edges_to_keep`, "
            "`fraction_primal_edges_to_keep` and `primal_att_coeff_threshold` "
            "must be non-None.")
        if (num_primal_edges_to_keep is not None):
            assert (isinstance(num_primal_edges_to_keep, int))
            assert (num_primal_edges_to_keep >= 0)
        if (fraction_primal_edges_to_keep is not None):
            assert (isinstance(fraction_primal_edges_to_keep, float))
            assert (
                0.0 <= fraction_primal_edges_to_keep <= 1.0
            ), "The fraction of edges to be pooled must be in the range [0, 1]."
        if (primal_att_coeff_threshold is not None):
            assert (isinstance(primal_att_coeff_threshold, float))
            assert (
                0.0 <= primal_att_coeff_threshold <= 1.0
            ), "Attention coefficients are expected to be in the range [0, 1]."
            assert (allow_pooling_consecutive_edges is True), (
                "Pooling of non-consecutive edges only is not available with "
                "threshold-based pooling.")
        assert (aggr_primal_features in ['mean', 'add']), (
            "Aggregation mode of primal features can be either 'mean' or 'add'."
        )
        assert (aggr_dual_features in ['mean', 'add']), (
            "Aggregation mode of primal features can be either 'mean' or 'add'."
        )

        self.__self_loops_in_output_dual_graph = self_loops_in_output_dual_graph
        self.__single_dual_nodes = single_dual_nodes
        self.__undirected_dual_edges = undirected_dual_edges
        self.__num_primal_edges_to_keep = num_primal_edges_to_keep
        self.__fraction_primal_edges_to_keep = fraction_primal_edges_to_keep
        self.__primal_att_coeff_threshold = primal_att_coeff_threshold
        self.__allow_pooling_consecutive_edges = allow_pooling_consecutive_edges
        self.__add_to_primal_att_coeffs = add_to_primal_att_coeffs
        self.__aggr_primal_features = aggr_primal_features
        self.__aggr_dual_features = aggr_dual_features
        self.__use_decreasing_attention_coefficient = (
            use_decreasing_attention_coefficient)
        self.__return_old_dual_node_to_new_dual_node = (
            return_old_dual_node_to_new_dual_node)
        self.__log_ratio_new_old_primal_nodes = log_ratio_new_old_primal_nodes
        self.__log_ratio_new_old_primal_edges = log_ratio_new_old_primal_edges

        super(DualPrimalEdgePooling, self).__init__()

    def forward(self, primal_graph_batch, dual_graph_batch,
                primal_edge_to_dual_node_idx_batch, primal_attention_coeffs):
        r"""Forward computation which optionally adds a value to the primal-edge
        attention coefficients and merges the edges.
        Note: only pooling based on coefficients from a single attention head is
        supported.

        Args:
            primal_graph_batch (torch_geometric.data.batch.Batch): Batch
                containing the input primal graphs on which the attention-driven
                edge-pooling operation should be applied.
            dual_graph_batch (torch_geometric.data.batch.Batch): Batch
                containing the input dual graphs associated to the primal graphs
                whose edges will be pooled.
            primal_edge_to_dual_node_idx_batch (dict): Dictionary representing
                the associations between primal-graph edges and dual-graph nodes
                in the batch.
            primal_attention_coeffs (torch.Tensor of shape
                :obj:`[num_primal_edges, num_heads]`, where `num_primal_edges`
                is the number of edges in the primal-graph batch and `num_heads`
                is the number of attention heads): Attention coefficients
                associated to the edges in the primal-edge graph.
                Note: element [i, j] corresponds to the attention coefficient
                over the i-th edge in `primal_graph_batch.edge_index`, according
                to the j-th attention head.

        Returns:
            primal_graph_batch (torch_geometric.data.batch.Batch): Output
                primal-graph batch after edge pooling.
            dual_graph_batch (torch_geometric.data.batch.Batch): Output
                dual-graph batch after edge pooling.
            primal_edge_to_dual_node_idx_batch (dict): Output dictionary
                representing the associations between primal-graph edges and
                dual-graph nodes in the batch after edge pooling.
            pooling_log (pd_mesh_net.nn.pool.PoolingInfo): Information that maps
                the original graphs with those graphs returned after pooling.
                Optionally, if the arguments `log_ratio_new_old_primal_nodes`/
                `log_ratio_new_old_primal_edges` are True, its elements with key
                `ratio_new_old_primal_nodes`/'ratio_new_old_primal_edges' are
                a list of `num_samples` elements - where `num_samples` is the
                number of samples in the batch - the i-th element of which
                contains the ratio between the number of primal nodes/edges in
                the new, pooled graph and the number of primal nodes/edges
                before pooling, for the i-th sample in the batch. If any of the
                two class arguments is False, the corresponding element is None.
        """
        assert (primal_attention_coeffs.dim() == 2)
        # Average attention coefficients over the heads.
        primal_attention_coeffs = primal_attention_coeffs.mean(axis=1)
        assert (primal_attention_coeffs.dim() == 1)
        # Optionally add a value to all the primal-edge attention coefficients.
        primal_attention_coeffs = (primal_attention_coeffs.view(-1) +
                                   self.__add_to_primal_att_coeffs)

        (primal_graph_batch, dual_graph_batch,
         primal_edge_to_dual_node_idx_batch,
         pooling_log) = self.__pool_primal_edges(
             primal_graph_batch, dual_graph_batch,
             primal_edge_to_dual_node_idx_batch, primal_attention_coeffs)

        return (primal_graph_batch, dual_graph_batch,
                primal_edge_to_dual_node_idx_batch, pooling_log)

    def __pool_primal_edges(self, primal_graph_batch, dual_graph_batch,
                            primal_edge_to_dual_node_idx_batch,
                            primal_attention_coeffs):
        r"""Pools primal-graph edges until either the predefined number of edges
        are left in the primal graphs in the batch (cf. class argument
        `num_primal_edges_to_keep`) or all the edges left have attention
        coefficient smaller/larger than a predefined value (cf. class argument
        `primal_att_coeff_threshold`), depending on the class argument
        `use_decreasing_attention_coefficient`. Modified from
        `torch_geometric.nn.pool.EdgePooling`.

        Args:
            primal_graph_batch (torch_geometric.data.batch.Batch): Batch
                containing the input primal graphs on which the attention-driven
                edge-pooling operation should be applied.
            dual_graph_batch (torch_geometric.data.batch.Batch): Batch
                containing the input dual graphs associated to the primal graphs
                whose edges will be pooled.
            primal_edge_to_dual_node_idx_batch (dict): Dictionary representing
                the associations between primal-graph edges and dual-graph nodes
                in the batch.
            primal_attention_coeffs (torch.Tensor of shape
                :obj:`[num_primal_edges,]`, where `num_primal_edges` is the
                number of edges in the primal-graph batch): Attention
                coefficients associated to the edges in the primal-edge graph.
                Note: the i-th element corresponds to the attention coefficient
                over the i-th edge in `primal_graph_batch.edge_index`.

        Returns:
            primal_graph_batch (torch_geometric.data.batch.Batch): Output
                primal-graph batch after edge pooling.
            dual_graph_batch (torch_geometric.data.batch.Batch): Output
                dual-graph batch after edge pooling.
            primal_edge_to_dual_node_idx_batch (dict): Output dictionary
                representing the associations between primal-graph edges and
                dual-graph nodes in the batch after edge pooling.
            pooling_log (pd_mesh_net.nn.pool.PoolingInfo): Information that maps
                the original graphs with those graphs returned after pooling.
                Optionally, if the arguments `log_ratio_new_old_primal_nodes`/
                `log_ratio_new_old_primal_edges` are True, its elements with key
                `ratio_new_old_primal_nodes`/'ratio_new_old_primal_edges' are
                a list of `num_samples` elements - where `num_samples` is the
                number of samples in the batch - the i-th element of which
                contains the ratio between the number of primal nodes/edges in
                the new, pooled graph and the number of primal nodes/edges
                before pooling, for the i-th sample in the batch. If any of the
                two class arguments is False, the corresponding element is None.
        """

        num_primal_nodes = primal_graph_batch.num_nodes
        num_primal_edges = primal_graph_batch.num_edges

        # New tensors need to be moved to GPU if the input tensors are on GPU.
        device = primal_graph_batch.edge_index.device

        # Sort the list of edges of the primal graphs.
        # Find the indices of the 'forward' primal edges, i.e., primal edges
        # i->j with i < j.
        is_edge_forward = (primal_graph_batch.edge_index[0, :] <
                           primal_graph_batch.edge_index[1, :])
        # - Order the forward edges increasingly, first by i and then by j.
        _, forward_direction_indices = coalesce(
            index=primal_graph_batch.edge_index[:, is_edge_forward],
            value=torch.arange(num_primal_edges,
                               dtype=torch.long,
                               device=device)[is_edge_forward],
            m=num_primal_nodes,
            n=num_primal_nodes)
        assert (forward_direction_indices.shape[0] == num_primal_edges //
                2), "Error: found two equal primal edges in the graph."
        # Find the indices of the 'backward' primal edges, i.e., primal edges
        # j->i with j > i.
        is_edge_backward = ~is_edge_forward
        _, backward_direction_indices = coalesce(
            index=primal_graph_batch.edge_index[:, is_edge_backward].flip(0),
            value=torch.arange(num_primal_edges,
                               dtype=torch.long,
                               device=device)[is_edge_backward],
            m=num_primal_nodes,
            n=num_primal_nodes)
        assert (backward_direction_indices.shape[0] == num_primal_edges //
                2), "Error: found two equal primal edges in the graph."
        # Average the primal attention coefficients over pairs of opposite
        # primal edges i->j and j->i.
        average_primal_attention_coeffs = (
            primal_attention_coeffs[forward_direction_indices] +
            primal_attention_coeffs[backward_direction_indices]) / 2

        if (self.__primal_att_coeff_threshold is not None):
            # Threshold-based pooling.
            # Find indices of the primal edges {i, j} for which the average of
            # the attention coefficients over i->j and j->i is larger/smaller
            # than the predefined threshold, depending on
            # self.__use_decreasing_attention_coefficient.
            assert (num_primal_edges % 2 == 0)
            if (self.__use_decreasing_attention_coefficient):
                should_sorted_edge_pair_be_pooled = (
                    average_primal_attention_coeffs >
                    self.__primal_att_coeff_threshold)
            else:
                should_sorted_edge_pair_be_pooled = (
                    average_primal_attention_coeffs <
                    self.__primal_att_coeff_threshold)
            forward_edges_to_pool = (
                primal_graph_batch.edge_index[:, forward_direction_indices]
                [:, should_sorted_edge_pair_be_pooled])
            backward_edges_to_pool = (
                primal_graph_batch.edge_index[:, backward_direction_indices]
                [:, should_sorted_edge_pair_be_pooled])
            assert (torch.equal(forward_edges_to_pool,
                                backward_edges_to_pool.flip(0)))
            del should_sorted_edge_pair_be_pooled
        else:
            # Top-k pooling.
            should_sorted_edge_be_pooled = torch.zeros(num_primal_edges,
                                                       dtype=torch.bool,
                                                       device=device)
            if (not self.__allow_pooling_consecutive_edges):
                was_edge_connected_to_this_primal_node_pooled = torch.zeros(
                    num_primal_nodes, dtype=torch.bool, device=device)
            for sample_idx in range(primal_graph_batch.num_graphs):
                # - For each sample in the batch, find the forward and backward
                #   primal edges that belong to that sample.
                is_node_in_sample = primal_graph_batch.batch == sample_idx
                is_forward_edge_in_sample = (is_node_in_sample)[
                    primal_graph_batch.edge_index[:,
                                                  forward_direction_indices][0]]
                is_backward_edge_in_sample = (
                    is_node_in_sample
                )[primal_graph_batch.edge_index[:,
                                                backward_direction_indices][0]]
                forward_direction_indices_in_sample = forward_direction_indices[
                    is_forward_edge_in_sample]
                backward_direction_indices_in_sample = (
                    backward_direction_indices[is_backward_edge_in_sample])

                assert (len(forward_direction_indices_in_sample) == len(
                    backward_direction_indices_in_sample))
                num_primal_edges_in_sample = len(
                    forward_direction_indices_in_sample)
                # - Find the number of edges to pool or, if given already,
                #   verify that there are enough edges that can be pooled.
                if (self.__fraction_primal_edges_to_keep is not None):
                    # Amount of primal edges to pool given as a fraction.
                    num_primal_edges_to_pool = int(
                        (1.0 - self.__fraction_primal_edges_to_keep) *
                        num_primal_edges_in_sample)
                else:
                    # Amount of primal edges to pool given as a number.
                    num_primal_edges_to_pool = (num_primal_edges_in_sample -
                                                self.__num_primal_edges_to_keep)
                    if (num_primal_edges_to_pool < 0):
                        # More edges to keep than edges in the sample. -> Do not
                        # perform pooling.
                        num_primal_edges_to_pool = 0

                # Find the top attention coefficients associated to each pair of
                # opposite primal edges i->j and j->i.
                average_primal_attention_coeffs = (
                    primal_attention_coeffs[forward_direction_indices_in_sample]
                    + primal_attention_coeffs[
                        backward_direction_indices_in_sample]) / 2

                # Select the k pair of opposite primal edges with the highest/
                # lowest attention coefficients (where k is the number defined
                # above), depending on
                # self.__use_decreasing_attention_coefficient.
                _, indices_pairs_to_select = torch.sort(
                    average_primal_attention_coeffs,
                    descending=self.__use_decreasing_attention_coefficient)
                if (self.__allow_pooling_consecutive_edges):
                    indices_pairs_to_select = (
                        indices_pairs_to_select[:num_primal_edges_to_pool])
                    indices_forward_edges_to_pool_in_sample = (
                        forward_direction_indices_in_sample[
                            indices_pairs_to_select])
                    should_sorted_edge_be_pooled[
                        indices_forward_edges_to_pool_in_sample] = True
                    # Also backward edges could be selected here, but it is not
                    # necessary, since the graph-update method only requires
                    # forward edges.
                else:
                    # Only select a primal edge for pooling if no other primal
                    # edges sharing a primal node with it have been pooled
                    # before.
                    num_edges_pooled_in_sample = 0
                    pooled_all_required_edges = False
                    for idx_current_edge_pair in indices_pairs_to_select:
                        if (num_edges_pooled_in_sample ==
                                num_primal_edges_to_pool):
                            pooled_all_required_edges = True
                            break
                        # - Find the primal nodes connected to this primal edge.
                        forward_direction_index_in_sample = (
                            forward_direction_indices_in_sample[
                                idx_current_edge_pair])
                        primal_nodes = (
                            primal_graph_batch.
                            edge_index[:, forward_direction_index_in_sample])
                        if (not was_edge_connected_to_this_primal_node_pooled[
                                primal_nodes].any()):
                            should_sorted_edge_be_pooled[
                                forward_direction_index_in_sample] = True
                            was_edge_connected_to_this_primal_node_pooled[
                                primal_nodes] = True
                            num_edges_pooled_in_sample += 1
                    assert (pooled_all_required_edges), (
                        "Unable to pool the requested number of "
                        "non-consecutive primal edges "
                        f"({num_primal_edges_to_pool}).")

            if (sample_idx > 0):
                del average_primal_attention_coeffs, indices_pairs_to_select
                if (self.__allow_pooling_consecutive_edges):
                    del indices_forward_edges_to_pool_in_sample

            # Select all the forward edges to pool.
            forward_edges_to_pool = (
                primal_graph_batch.edge_index[:, should_sorted_edge_be_pooled])

            del should_sorted_edge_be_pooled
            if (not self.__allow_pooling_consecutive_edges):
                del was_edge_connected_to_this_primal_node_pooled

        # Perform the actual pooling and update the graphs.
        (new_primal_graph_batch, new_dual_graph_batch, new_petdni_graph,
         node_to_cluster,
         old_dual_node_to_new_dual_node) = self.__update_graphs__(
             forward_primal_edges_to_pool=forward_edges_to_pool,
             primal_graph_batch=primal_graph_batch,
             dual_graph_batch=dual_graph_batch,
             primal_edge_to_dual_node_idx_batch=
             primal_edge_to_dual_node_idx_batch,
             num_primal_nodes=num_primal_nodes,
             num_primal_edges=num_primal_edges,
             device=device)

        # Return information associated to the operations performed while
        # pooling.
        old_primal_node_to_new_one = node_to_cluster

        # If required, for each sample in the batch, return the ratio between
        # the number of primal nodes/edges in the new, pooled graph and the
        # number of primal nodes/edges before pooling.
        ratio_new_old_nodes = None
        ratio_new_old_edges = None
        if (self.__log_ratio_new_old_primal_nodes):
            # - Find the sample to which each old node and each new node belong.
            num_new_nodes_per_sample = (
                new_primal_graph_batch.batch.unique_consecutive(
                    return_counts=True))[1].type(torch.float)
            num_old_nodes_per_sample = (
                primal_graph_batch.batch.unique_consecutive(
                    return_counts=True))[1]
            ratio_new_old_nodes = (num_new_nodes_per_sample /
                                   num_old_nodes_per_sample)

            del num_new_nodes_per_sample, num_old_nodes_per_sample
        if (self.__log_ratio_new_old_primal_nodes):
            # - Find the sample to which each old node and each new edge belong.
            num_new_edges_per_sample = (new_primal_graph_batch.batch[
                new_primal_graph_batch.edge_index[0]].unique_consecutive(
                    return_counts=True))[1].type(torch.float)
            num_old_edges_per_sample = (primal_graph_batch.batch[
                primal_graph_batch.edge_index[0]].unique_consecutive(
                    return_counts=True))[1]
            ratio_new_old_edges = (num_new_edges_per_sample /
                                   num_old_edges_per_sample)

            del num_new_edges_per_sample, num_old_edges_per_sample

        pooling_log = PoolingInfo(
            old_primal_node_to_new_one=old_primal_node_to_new_one,
            old_dual_node_to_new_one=old_dual_node_to_new_dual_node,
            old_primal_edge_index=primal_graph_batch.edge_index,
            old_dual_edge_index=dual_graph_batch.edge_index,
            old_primal_graph_batch=primal_graph_batch.batch,
            old_dual_graph_batch=dual_graph_batch.batch,
            old_primal_edge_to_dual_node_index=
            primal_edge_to_dual_node_idx_batch,
            ratio_new_old_primal_edges=ratio_new_old_edges,
            ratio_new_old_primal_nodes=ratio_new_old_nodes)

        return (new_primal_graph_batch, new_dual_graph_batch, new_petdni_graph,
                pooling_log)

    def __update_graphs__(self, forward_primal_edges_to_pool,
                          primal_graph_batch, dual_graph_batch,
                          primal_edge_to_dual_node_idx_batch, num_primal_nodes,
                          num_primal_edges, device):
        r"""Performs pooling of the input primal edges, and updates the primal
        and dual graph accordingly. In particular, primal nodes are merged and
        the dual graph is recreated according to the newly-formed primal-node
        'clusters'. Whenever two nodes are merged, their features are averaged.
        #TODO: Document better.

        Args:
            forward_primal_edges_to_pool (torch.Tensor of shape
                :obj:`[2, num_forward_primal_edges_to_pool]`, where
                `num_forward_primal_edges_to_pool` is the number of edges to
                pool): Forward primal edges that should be pooled, in the same
                format as the edge-index matrix. 'Forward' stands for the fact
                that, for every valid `i`,`forward_primal_edges_to_pool[0, i]`
                is smaller than `forward_primal_edges_to_pool[1, i]`.
            primal_graph_batch (torch_geometric.data.batch.Batch): Batch
                containing the input primal graphs on which the attention-driven
                edge-pooling operation should be applied.
            dual_graph_batch (torch_geometric.data.batch.Batch): Batch
                containing the input dual graphs associated to the primal graphs
                whose edges will be pooled.
            primal_edge_to_dual_node_idx_batch (dict): Dictionary representing
                the associations between primal-graph edges and dual-graph nodes
                in the batch.
            num_primal_nodes (int): Number of primal nodes in the input batch.
            num_primal_edges (int): Number of primal edges in the input batch.
            device (torch.device): Type of device on which the tensors created
                within this method should be allocated. (default: :obj:`None`)

        Returns:
            new_primal_graph_batch (torch_geometric.data.batch.Batch): Batch
                containing the primal graphs after pooling.
            new_dual_graph_batch (torch_geometric.data.batch.Batch): Batch
                containing the dual graphs after pooling.
            new_petdni_graph (dict): Dictionary representing the associations
                between primal-graph edges and dual-graph nodes in the batch
                after pooling.
            node_to_cluster (torch.Tensor of shape
                :obj:`[num_input_primal_nodes,]`, where `num_input_primal_nodes`
                is the number of primal nodes in the input batch): The `i`-th
                element contains the index of the primal node in the output
                batch to which the `i`-th primal node in the input graph
                corresponds (i.e., the 'cluster' in which it was merged, if
                any).
            old_dual_node_to_new_dual_node (torch.Tensor of shape
                :obj:`[num_input_dual_nodes,]` where `num_input_dual_nodes` is
                the number of dual nodes in the input batch/None): If the class
                argument `return_old_dual_node_to_new_dual_node` is False, None.
                Otherwise, tensor the i-th element of which maps the i-th input
                dual node to its corresponding output dual node; if the i-th
                input dual node has no corresponding output dual node, it is
                mapped to -1.
        """
        # - Convert the primal-edge-to-dual-node-idx dictionary to a tensor.
        #   TODO: This step can be avoided by using tensors in the
        #   first place.
        petdni_list = []
        for edge in primal_graph_batch.edge_index.t().tolist():
            petdni_list.append(primal_edge_to_dual_node_idx_batch[tuple(edge)])

        petdni_tensor = torch.tensor(petdni_list,
                                     dtype=torch.long,
                                     device=device)
        # 'Cluster' together all the edges to merge that share an endpoint. All
        # the primal nodes that are endpoints of one edge in a same cluster will
        # be merged together.
        node_cluster_creator = NodeClustersWithUnionFind(
            will_input_tensors=False)
        for edge in forward_primal_edges_to_pool.t().tolist():
            # Add the endpoints of an edge to a node cluster if at least one of
            # the two endpoints is in the set of nodes in the cluster.
            node_cluster_creator.add_nodes_from_edge(edge)

        merged_node_clusters = node_cluster_creator.clusters

        # Map all nodes in the primal graph to the node-cluster to which they
        # belong (which coincides with their initial index by default).
        node_to_cluster = torch.arange(num_primal_nodes, device=device)

        if (self.__single_dual_nodes):
            # Configuration A.
            num_dual_nodes = num_primal_edges // 2
        else:
            # Configurations B and C.
            num_dual_nodes = num_primal_edges

        if (len(merged_node_clusters) == 0):
            # - When using threshold-based pooling, it might happen that no
            #   primal edges have an associated attention coefficient larger
            #   than the threshold, hence no primal edges can be pooled. If this
            #   is the case leave the input graphs unvaried.
            print(f"\033[93mWarning: no primal edges are being pooled.\033[00m")
            new_primal_graph_batch = primal_graph_batch
            new_dual_graph_batch = dual_graph_batch
            new_petdni_graph = primal_edge_to_dual_node_idx_batch
            old_dual_node_to_new_dual_node = torch.arange(num_dual_nodes,
                                                          dtype=torch.long,
                                                          device=device)

            return (new_primal_graph_batch, new_dual_graph_batch,
                    new_petdni_graph, node_to_cluster,
                    old_dual_node_to_new_dual_node)

        # For each cluster of primal nodes being merged/primal edges being
        # pooled:
        merged_node_clusters_as_tensors = [
            torch.tensor(list(merged_node_cluster),
                         dtype=torch.long,
                         device=device)
            for merged_node_cluster in merged_node_clusters
        ]
        all_nodes_in_clusters = torch.cat(
            [cluster for cluster in merged_node_clusters_as_tensors])
        # - Merge all the nodes in the cluster to the first node in the cluster.
        node_to_representative_cluster_node = torch.cat([
            torch.full(cluster.size(),
                       cluster[0],
                       dtype=torch.long,
                       device=device)
            for cluster in merged_node_clusters_as_tensors
        ])
        representative_cluster_nodes = torch.tensor(
            [cluster[0] for cluster in merged_node_clusters_as_tensors],
            dtype=torch.long,
            device=device)
        node_to_cluster[
            all_nodes_in_clusters] = node_to_representative_cluster_node

        del merged_node_clusters_as_tensors

        # - Update the feature of the representative primal node in the cluster
        #   with the average/sum of the features of the nodes in the cluster.
        features_to_average = primal_graph_batch.x[all_nodes_in_clusters]
        new_primal_features = primal_graph_batch.x.clone()
        if (self.__aggr_primal_features == 'mean'):
            new_primal_features[representative_cluster_nodes] = scatter_mean(
                features_to_average.t(), node_to_representative_cluster_node).t(
                )[representative_cluster_nodes]
        else:
            new_primal_features[representative_cluster_nodes] = scatter_add(
                features_to_average.t(), node_to_representative_cluster_node).t(
                )[representative_cluster_nodes]

        del all_nodes_in_clusters, features_to_average
        del node_to_representative_cluster_node, representative_cluster_nodes
        # Reassign indices to the nodes so that all indices between 0 and the
        # new number of primal nodes - 1 have a corresponding primal node.
        representative_nodes, node_to_cluster = node_to_cluster.unique(
            return_inverse=True)
        assert (len(node_to_cluster) == num_primal_nodes)
        new_primal_features = new_primal_features[representative_nodes]
        # Define the new edge-index matrix.
        new_primal_edge_index = node_to_cluster[primal_graph_batch.edge_index]
        num_primal_nodes_after_pooling = len(representative_nodes)
        # Redefine the indexing of the primal edges, their correspondence with
        # dual nodes, and the dual features in the following way:
        # - Remove duplicate edges between newly-formed primal-node clusters
        #   (here actually no effect is applied on the edge-index tensor, as the
        #   operation will be repeated below). While doing this, update the
        #   correspondence between primal edges and dual nodes. Note: when two
        #   clusters \mathcal{A} and \mathcal{B} of faces share more than one
        #   edge (and therefore new_primal_edge_index contains multiple edges
        #   \mathcal{A}->\mathcal{B} / \mathcal{B}->\mathcal{A}), there is more
        #   than one node in the old dual graph that corresponds to the new dual
        #   node \mathcal{A}->\mathcal{B} (all those of the form P->Q, with P in
        #   \mathcal{A} and Q in \mathcal{B}, considering WLOG configurations B
        #   and C). Of the multiple dual nodes, we select WLOG the one with the
        #   highest index. Note: this is mathematically sound, as the feature of
        #   the new dual node will be set so as to correspond to the average/
        #   sum of the multiple dual nodes, cf. below.
        _, new_petdni_tensor = coalesce(new_primal_edge_index,
                                        petdni_tensor,
                                        num_primal_nodes_after_pooling,
                                        num_primal_nodes_after_pooling,
                                        op='max')

        if (self.__return_old_dual_node_to_new_dual_node):
            # - Map each dual node to the new primal-edge index that will be
            #   obtained after removing duplicates (i.e., coalescing, see
            #   below).
            old_dual_node_to_new_primal_edge_index = torch.empty(
                num_dual_nodes, dtype=torch.long, device=device)
            old_primal_edge_index_to_new_primal_edge_index = torch.unique(
                new_primal_edge_index, return_inverse=True, dim=1)[1]
            old_dual_node_to_new_primal_edge_index[petdni_tensor] = (
                old_primal_edge_index_to_new_primal_edge_index)

            del old_primal_edge_index_to_new_primal_edge_index
        # - Set the features of the new dual nodes to correspond to the average/
        #   sum of the features of all the 'old' nodes associated to the edges
        #   between new primal-node clusters (cf. above).
        new_primal_edge_index_with_self_loops, new_dual_features = coalesce(
            new_primal_edge_index,
            dual_graph_batch.x[petdni_tensor],
            num_primal_nodes_after_pooling,
            num_primal_nodes_after_pooling,
            op=self.__aggr_dual_features)

        del petdni_tensor
        # - Since merging faces causes primal nodes to end up in the same
        #   cluster, remove edges between each newly-formed cluster and itself,
        #   and consequently update the indexing of the correspondences between
        #   primal edges and dual nodes and the dual features.
        #TODO: The following three `remove_self_loops` operations can
        # possibly be performed in a single instruction by stacking the
        # attribute tensors.
        _, new_petdni_tensor = remove_self_loops(
            new_primal_edge_index_with_self_loops, new_petdni_tensor)
        new_primal_edge_index, new_dual_features = remove_self_loops(
            new_primal_edge_index_with_self_loops, new_dual_features)
        if (self.__return_old_dual_node_to_new_dual_node):
            _, indices_primal_edges_kept = remove_self_loops(
                new_primal_edge_index_with_self_loops,
                torch.arange(new_primal_edge_index_with_self_loops.shape[1],
                             dtype=torch.long,
                             device=device))

            # - Update the maps from dual nodes to new primal-edge indices after
            #   removal of self-loops in the primal graph. Dual nodes that are
            #   not mapped to any new primal-edge are mapped to a value of -1.
            npei_with_self_loops_to_npei = -torch.ones_like(
                new_primal_edge_index_with_self_loops[0])
            npei_with_self_loops_to_npei[
                indices_primal_edges_kept] = torch.arange(
                    len(indices_primal_edges_kept),
                    dtype=torch.long,
                    device=device)
            old_dual_node_to_new_primal_edge_index = (
                npei_with_self_loops_to_npei[
                    old_dual_node_to_new_primal_edge_index])

            del indices_primal_edges_kept, npei_with_self_loops_to_npei

        if (self.__single_dual_nodes):
            # Opposite edges correspond to the same dual node, i.e., dual nodes
            # are single. Configuration A. One needs to remove the dual features
            # associated to 'backward' primal edges.
            is_new_primal_edge_forward = (
                new_primal_edge_index == new_primal_edge_index.t().sort()
                [0].t()).all(0)
            new_dual_features = new_dual_features[is_new_primal_edge_forward]

        # Define the new primal-node-to-batch-sample correspondence.
        new_primal_batch = primal_graph_batch.batch[representative_nodes]

        # Reconstruct the dual graphs.
        # - Find the dual nodes to keep.
        indices_dual_nodes_to_keep = new_petdni_tensor.unique()
        # - Define the new dual-node-to-batch-sample correspondence.
        new_dual_batch = dual_graph_batch.batch[indices_dual_nodes_to_keep]
        # - Reconstruct the dual edges.
        #   - Instantiate first the new edge-index matrix to contain twice as
        #     many edges as the initial dual-graph batch.
        new_dual_edge_index = (2 * dual_graph_batch.num_edges) * [None]
        num_primal_edges_after_pooling = new_primal_edge_index.shape[1]
        if (self.__single_dual_nodes):
            # Dual-graph configuration A.
            # - Take only the 'forward' ones among the new primal edges.
            indices_primal_edges_to_consider = torch.arange(
                num_primal_edges_after_pooling,
                device=device)[is_new_primal_edge_forward]
            (primal_edges_to_consider
            ) = new_primal_edge_index[:, indices_primal_edges_to_consider]
        else:
            # - Take all the new primal edges.
            primal_edges_to_consider = new_primal_edge_index

        primal_edges_to_consider_list = primal_edges_to_consider.t().tolist()

        del primal_edges_to_consider
        if (self.__single_dual_nodes):
            new_petdni_edges_to_consider_list = new_petdni_tensor[
                indices_primal_edges_to_consider].tolist()

            del indices_primal_edges_to_consider
            # Dual-graph configuration A.
            # For each primal node \mathcal{A}, find the indices of all the dual
            # nodes {\mathcal{A}, \mathcal{B}}, i.e., of those associated to a
            # primal edge that has \mathcal{A} as an endpoint.
            primal_node_to_dual_nodes = [
                set() for _ in range(num_primal_nodes_after_pooling)
            ]
            for primal_edge, dual_node in zip(
                    primal_edges_to_consider_list,
                    new_petdni_edges_to_consider_list):
                primal_node_to_dual_nodes[primal_edge[0]].add(dual_node)
                primal_node_to_dual_nodes[primal_edge[1]].add(dual_node)
            # For each primal node \mathcal{A}, connect the dual nodes
            # {\mathcal{A}, \mathcal{B}} to all dual nodes
            # {\mathcal{A}, \mathcal{M}} with a bidirectional edge, optionally
            # allowing \mathcal{M} = \mathcal{B}. Note: by iterating over all
            # \mathcal{A}, the dual nodes {\mathcal{A}, \mathcal{B}} will
            # automatically be connected also to the dual nodes
            # {\mathcal{B}, \mathcal{N}}.
            num_new_dual_edges = 0
            if (self.__self_loops_in_output_dual_graph):
                dual_nodes_with_added_self_loop = set()
            # Iterate on the primal nodes \mathcal{A}.
            for primal_node in range(num_primal_nodes_after_pooling):
                primal_node_to_dual_nodes[primal_node] = list(
                    primal_node_to_dual_nodes[primal_node])
                num_dual_nodes_with_current_primal_node = len(
                    primal_node_to_dual_nodes[primal_node])

                for dual_node_local_idx in range(
                        num_dual_nodes_with_current_primal_node):
                    dual_node = primal_node_to_dual_nodes[primal_node][
                        dual_node_local_idx]
                    if (self.__self_loops_in_output_dual_graph):
                        if (not dual_node in dual_nodes_with_added_self_loop):
                            # Dual edge {\mathcal{A}, \mathcal{B}}->
                            # {\mathcal{A}, \mathcal{B}}.
                            new_dual_edge_index[num_new_dual_edges] = [
                                dual_node, dual_node
                            ]
                            num_new_dual_edges += 1
                            dual_nodes_with_added_self_loop.add(dual_node)

                    # Dual nodes {\mathcal{A}, \mathcal{M}}.
                    for other_dual_node_local_idx in range(
                            dual_node_local_idx + 1,
                            num_dual_nodes_with_current_primal_node):
                        other_dual_node = primal_node_to_dual_nodes[
                            primal_node][other_dual_node_local_idx]
                        # - Dual edges {\mathcal{A}, \mathcal{B}}->
                        #   {\mathcal{A}, \mathcal{M}}.
                        new_dual_edge_index[num_new_dual_edges] = [
                            dual_node, other_dual_node
                        ]
                        # - Dual edges {\mathcal{A}, \mathcal{M}}->
                        #   {\mathcal{A}, \mathcal{B}}.
                        new_dual_edge_index[num_new_dual_edges +
                                            1] = [other_dual_node, dual_node]
                        num_new_dual_edges += 2
        else:
            # Dual-graph configurations B and C.
            new_petdni_edges_to_consider_list = new_petdni_tensor.tolist()
            # For each primal node \mathcal{A}, find the indices of all the dual
            # nodes \mathcal{A}->\mathcal{B}, i.e., of those associated to a
            # primal edge outgoing from \mathcal{A}, and of all the dual nodes
            # \mathcal{B}->\mathcal{A}, i.e., of those associated to a primal
            # edge incoming in \mathcal{A}.
            primal_node_to_incoming_primal_edges = [
                set() for _ in range(num_primal_nodes_after_pooling)
            ]
            primal_node_to_outgoing_primal_edges = [
                set() for _ in range(num_primal_nodes_after_pooling)
            ]
            # Store, for each dual node, the index of its corresponding primal
            # edge. This will be used, for each dual node, to avoid connecting
            # it to its opposite dual node.
            dual_node_to_primal_edge_idx = dict()
            for primal_edge_idx, (primal_edge, dual_node) in enumerate(
                    zip(primal_edges_to_consider_list,
                        new_petdni_edges_to_consider_list)):
                primal_node_to_outgoing_primal_edges[primal_edge[0]].add(
                    dual_node)
                primal_node_to_incoming_primal_edges[primal_edge[1]].add(
                    dual_node)
                dual_node_to_primal_edge_idx[dual_node] = primal_edge_idx
            # For each primal node \mathcal{A}, connect the dual nodes
            # \mathcal{A}->\mathcal{B} to all dual nodes
            # \mathcal{M}->\mathcal{A} with a bidirectional edge (in case of
            # dual-graph configuration B), or only with an incoming edge (in
            # case of dual-graph configuration C), optionally adding self-loops.
            # Note: by iterating over all \mathcal{A}, the dual nodes
            # \mathcal{A}->\mathcal{B} will automatically be connected also to
            # the dual nodes \mathcal{B}->\mathcal{N}.
            num_new_dual_edges = 0
            # Iterate on the primal nodes \mathcal{A}.
            for primal_node in range(num_primal_nodes_after_pooling):
                num_dual_nodes_with_current_primal_node = len(
                    primal_node_to_outgoing_primal_edges[primal_node])
                for dual_node in primal_node_to_outgoing_primal_edges[
                        primal_node]:
                    if (self.__self_loops_in_output_dual_graph):
                        # Dual edge (\mathcal{A}->\mathcal{B})->
                        # (\mathcal{A}->\mathcal{B}).
                        new_dual_edge_index[num_new_dual_edges] = [
                            dual_node, dual_node
                        ]
                        num_new_dual_edges += 1

                    # Dual nodes \mathcal{M}->\mathcal{A}.
                    for other_dual_node in (
                            primal_node_to_incoming_primal_edges[primal_node]):
                        # Check that \mathcal{M} != \mathcal{B}.
                        current_primal_edge = primal_edges_to_consider_list[
                            dual_node_to_primal_edge_idx[dual_node]]
                        other_primal_edge = primal_edges_to_consider_list[
                            dual_node_to_primal_edge_idx[other_dual_node]]
                        if (current_primal_edge == other_primal_edge[::-1]):
                            continue
                        # - Dual edges (\mathcal{M}->\mathcal{A})->
                        #   (\mathcal{A}->\mathcal{B}).
                        new_dual_edge_index[num_new_dual_edges] = [
                            other_dual_node, dual_node
                        ]
                        if (self.__undirected_dual_edges):
                            # Dual-graph configuration B.
                            # - Dual edges (\mathcal{A}->\mathcal{B})->
                            #   (\mathcal{M}->\mathcal{A}).
                            new_dual_edge_index[num_new_dual_edges + 1] = [
                                dual_node, other_dual_node
                            ]
                            num_new_dual_edges += 2
                        else:
                            # Dual-graph configuration C.
                            num_new_dual_edges += 1

        new_dual_edge_index = torch.tensor(
            new_dual_edge_index[:num_new_dual_edges],
            dtype=torch.long,
            device=device).t()

        # - Remap the new dual nodes to range from 0 to the number of new dual
        #   nodes.
        representative_old_dual_node_to_new_dual_node = torch.empty(
            num_dual_nodes, dtype=torch.long, device=device)
        new_dual_node_indices = torch.arange(len(indices_dual_nodes_to_keep),
                                             dtype=torch.long,
                                             device=device)
        representative_old_dual_node_to_new_dual_node[
            indices_dual_nodes_to_keep] = new_dual_node_indices

        del indices_dual_nodes_to_keep, new_dual_node_indices
        # - Remap correspondences between primal edges and dual node indices
        #   accordingly.
        new_petdni_tensor = representative_old_dual_node_to_new_dual_node[
            new_petdni_tensor]
        # - Remap the dual edges accordingly.
        new_dual_edge_index = representative_old_dual_node_to_new_dual_node[
            new_dual_edge_index]

        del representative_old_dual_node_to_new_dual_node
        if (self.__return_old_dual_node_to_new_dual_node):
            # - For the purpose of reverting the pooling operation (when
            #   performing unpooling), map all the original dual nodes to their
            #   corresponding new dual nodes, mapping those that do not
            #   correspond to any new dual node to -1. Note: w.r.t.
            #   representative_old_dual_node_to_new_dual_node, here all the
            #   original dual nodes that get merged in one new dual node are
            #   mapped, not just the representative one.
            old_dual_node_to_new_dual_node = (
                old_dual_node_to_new_primal_edge_index.clone())
            old_dual_node_to_new_dual_node[
                old_dual_node_to_new_dual_node != -1] = new_petdni_tensor[
                    old_dual_node_to_new_dual_node[
                        old_dual_node_to_new_dual_node != -1]]
            del old_dual_node_to_new_primal_edge_index
        else:
            old_dual_node_to_new_dual_node = None

        # - The dual features until now are sorted in the feature matrix so as
        #   to match the order of the primal edges. They should instead be
        #   sorted so that their index in the feature matrix matches the index
        #   of the dual node to which they are associated.
        if (self.__single_dual_nodes):
            # Configuration A.
            new_dual_features[new_petdni_tensor[
                is_new_primal_edge_forward]] = new_dual_features.clone()
        else:
            # Configurations B and C.
            new_dual_features[new_petdni_tensor] = new_dual_features.clone()

        # Recreate the graphs.
        new_primal_graph_batch = Batch(batch=new_primal_batch)
        new_primal_graph_batch.edge_index = new_primal_edge_index
        new_primal_graph_batch.x = new_primal_features
        new_dual_graph_batch = Batch(batch=new_dual_batch)
        new_dual_graph_batch.edge_index = new_dual_edge_index
        new_dual_graph_batch.x = new_dual_features
        # TODO: this conversion step is not necessary anymore if petdni is a
        # tensor.
        new_petdni_graph = dict()
        new_primal_edge_index_list = new_primal_edge_index.t().tolist()
        new_petdni_tensor_list = new_petdni_tensor.tolist()
        assert (len(new_primal_edge_index_list) == len(new_petdni_tensor_list))
        for new_primal_edge, new_dual_node_idx in zip(
                new_primal_edge_index_list, new_petdni_tensor_list):
            new_petdni_graph[tuple(new_primal_edge)] = new_dual_node_idx

        return (new_primal_graph_batch, new_dual_graph_batch, new_petdni_graph,
                node_to_cluster, old_dual_node_to_new_dual_node)
