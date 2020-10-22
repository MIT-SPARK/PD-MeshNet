from torch.nn import GroupNorm, Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import avg_pool_x

from pd_mesh_net.models import BaseDualPrimalModel, DualPrimalResDownConv
from pd_mesh_net.nn import (GATConvNoSelfLoops, PoolingLogMultipleLayers)


class DualPrimalMeshClassifier(BaseDualPrimalModel):
    r"""Architecture for learning a global shape descriptor (classification).
    Based on analogous network from MeshCNN
    (https://github.com/ranahanocka/MeshCNN/).


    Args:
        in_channels_primal, in_channels_dual (int): Size of each input sample
            from the primal/dual graph respectively.
        norm_layer_type (str or None): Type of normalization layer to be used
            after each convolutional layer, both primal and dual. Possible
            options are: `None` (no normalization layer), `'group_norm'` (group
            normalization), `'batch_norm'` (batch normalization).
        num_groups_norm_layer (int): If `norm_layer_type` is 'group_norm',
            number of groups in each normalization layer.
        conv_primal_out_res, conv_dual_out_res (list of int): Number of output
            channels of each primal/dual convolutional layer respectively, i.e.,
            dimension of the feature vectors outputted by the primal/dual
            convolutional layers respectively.
        num_classes (int): Number of classes in the dataset.
        num_output_units_fc (int): Number of output unit of the first
            fully-connected layer (`fc1`).
        single_dual_nodes (bool): If True, it will be assumed that the dual
            graphs have single nodes; otherwise, it will be assumed that they
            have double nodes. Cf. :obj:`pd_mesh_net.utils.GraphCreator`.
        undirected_dual_edges (bool): If True, it will be assumed that every
            directed edge in the dual graphs has an opposite directed edge;
            otherwise, it will be assumed that directed edges in the dual graphs
            do not have an opposite directed edge.
            Cf. :obj:`pd_mesh_net.utils.GraphCreator`.
        num_primal_edges_to_keep (list of int/None, optional):
            Either None or a list of int values.
            - If not None, the value of the i-th element represents the target
              number of primal edges to keep in the pooling layer after the i-th
              convolutional layer.
            At most one of this and the arguments
            `fractions_primal_edges_to_keep` and
            `primal_attention_coeffs_thresholds` can be not None. If they all
            are None, no pooling layers are inserted in the network.
            (default: :obj:`None`)
        fractions_primal_edges_to_keep (list of float/None, optional):
            Either None or a list of float values.
            - If not None, the value of the i-th element represents the target
              fraction of primal edges to keep in the pooling layer after the
              i-th convolutional layer.
            At most one of this and the arguments `num_primal_edges_to_keep` and
            `primal_attention_coeffs_thresholds` can be not None. If they all
            are None, no pooling layers are inserted in the network.
            (default: :obj:`None`)
        primal_attention_coeffs_thresholds (list of float/None, optional):
            Either None or a list of float values.
            - If not None, the value of the i-th element represents the
              threshold used in the pooling layer after the i-th convolutional
              layer to pool primal edges for which the attention coefficient is
              above/below this value, depending on the argument
              `use_decreasing_attention_coefficients`.
            At most one of this and the arguments `num_primal_edges_to_keep` and
            `fractions_primal_edges_to_keep` can be not None. If they all are
            None, no pooling layers are inserted in the network.
            (default: :obj:`None`)
        num_res_blocks (int, optional): Number of residual blocks.
            (default: :obj:`3`).
        heads (int, optional): Number of attention heads associated to each of
            the primal and dual convolutional layers. (default: :obj:`1`)
        concat_primal, concat_dual (bool, optional): If set to :obj:`False`, the
            attention heads associated respectively to the primal and to the
            dual convolutional layers are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope_primal, negative_slope_dual (float, optional): LeakyReLU
            angle of the negative slope, respectively for the layers associated
            to the primal graph and for the layers associated to the dual graph.
            (default: :obj:`0.2`)
        dropout_primal, dropout_dual (float, optional): Dropout probability of
            the normalized attention coefficients which exposes each node,
            respectively of the primal and of the dual convolutional layers, to
            a stochastically sampled neighborhood during training.
            (default: :obj:`0`)
        bias_primal, bias_dual (bool, optional): If set to :obj:`False`, the
            layers associated respectively to the primal and to the dual graph
            will not learn an additive bias. (default: :obj:`False`)
        add_self_loops_to_dual_graph (bool, optional): If set to :obj:`True`,
            regular graph-attention convolutional layers are instantiated as
            dual convolutional layers, thus self-loops are added to the dual
            graph. Furthermore, self-loops are added to dual graphs generated by
            pooling layers. If set to :obj:`False` instead, a modified version
            of the graph-attention convolutional layer is instantiated for the
            dual convolutional layers, with no addition of self-loops to the
            latter, and no self loops are added to the dual graphs generated by
            pooling layers. (default: :obj:`False`)
        allow_pooling_consecutive_edges (bool, optional): If True, no
            restrictions are put on the primal edges that can be pooled by the
            pooling layers in the model. If False, a primal edge can only be
            pooled if no primal nodes to which it belongs have been pooled
            previously. Setting this argument to False is only compatible with
            top-K pooling (cf. arguments `num_primal_edges_to_keep` and
            `fractions_primal_edges_to_keep`). (default: :obj:`True`)
        aggr_primal_features_pooling (str, optional): Parameter of the optional
            pooling layer. If 'mean'/'add', the feature of each new primal node
            after pooling is obtained by respectively averaging and summing the
            features of the primal nodes that get merged into that node.
            (default: :obj:`mean`)
        aggr_dual_features_pooling (str, optional): Parameter of the optional
            pooling layer. If 'mean'/'add', whenever a new dual node is obtained
            by aggregating multiple previous dual nodes its feature is obtained
            by respectively averaging and summing the features of the previous
            dual nodes. (default: :obj:`mean`)
        use_decreasing_attention_coefficients (bool, optional): When using
            pooling based on an amount of edges (cf. arguments 
            `num_primal_edges_to_keep` and `fractions_primals_edges_to_keep`):
            if True, primal edges are pooled by decreasing magnitude of the
            associated attention coefficients; if False, primal edges are pooled
            by increasing magnitude of the associated attention coefficients.
            When using pooling based on attention-coefficient threshold (cf.
            argument `primal_attention_coeffs_thresholds`): primal edges are
            pooled if the associated attention coefficients are above the
            threshold - if True - or below the threshold - if False.
            (default: :obj:`True`)
        return_node_to_cluster (bool, optional): If True, each call to the
            forward method will also return a dict of `num_pooling_layers`
            lists, where `num_pooling_layers` is the number of pooling layers
            in the network. The i-th element of the list with key j will contain
            the index of the face cluster to which the i-th face in the mesh
            inputted to the j-th pooling layer belongs after the j-th pooling
            operation. If no pooling layer is present in the network, None is
            returned. (default: :obj:`False`)
        log_ratios_new_old_primal_nodes (bool, optional): If True, each call to
            the forward method will also return a namedtuple. Its element with
            key `ratios_new_old_primal_nodes` is a dict of `num_pooling_layers`
            tensors, where `num_pooling_layers` is the number of pooling layers
            in the network. Each of the `num_pooling_layers` tensors will be of
            length `num_samples` elements - where `num_samples` is the number of
            samples in the batch. The i-th element of the tensor with key j will
            contain the ratio between the number of primal nodes in the new,
            pooled graph and the number of primal nodes before pooling, for the
            i-th sample in the batch and with pooling applied by the pooling
            layer that follows the j-th convolutional layer in the network.
            (default: :obj:`True`)
        log_ratios_new_old_primal_edges (bool, optional): If True, each call to
            the forward method will also return a namedtuple. Its element with
            key `ratios_new_old_primal_edges` is a dict of `num_pooling_layers`
            tensors, where `num_pooling_layers` is the number of pooling layers
            in the network. Each of the `num_pooling_layers` tensors will be of
            length `num_samples` elements - where `num_samples` is the number of
            samples in the batch. The i-th element of the tensor with key j will
            contain the ratio between the number of primal edges in the new,
            pooled graph and the number of primal edges before pooling, for the
            i-th sample in the batch and with pooling applied by the pooling
            layer that follows the j-th convolutional layer in the network.
            (default: :obj:`False`)

    Attributes:
        fc1, fc2 (torch.nn.Linear): Fully-connected layers.
        _indices_pooling_layers (list): If the element `i` is in this list, then
            the i-th convolutional layer in the network is followed by a pooling
            layer.
        ---
        block{i} (pd_mesh_net.models.down_conv.DualPrimalResDownConv), `i` in
            `{0, ..., num_conv_layers - 1}`, `num_conv_layers` is defined based
            on the length of `conv_primal_out_res`: Dual-primal residual
            down-convolutional block, consisting of a dual-primal
            mesh-convolutional layer with output channels determined by
            `conv_primal/dual_out_res[i]`, an optional batch-/group-
            normalization layer, and an optional pooling layer with output
            resolution determined by the `i`-th element of
            `num_primal_edges_to_keep`, `fractions_primal_edges_to_keep` and
            `primal_attention_coeffs_thresholds`.
    """

    def __init__(self,
                 in_channels_primal,
                 in_channels_dual,
                 norm_layer_type,
                 num_groups_norm_layer,
                 conv_primal_out_res,
                 conv_dual_out_res,
                 num_classes,
                 num_output_units_fc,
                 single_dual_nodes,
                 undirected_dual_edges,
                 num_primal_edges_to_keep=None,
                 fractions_primal_edges_to_keep=None,
                 primal_attention_coeffs_thresholds=None,
                 num_res_blocks=3,
                 heads=1,
                 concat_primal=True,
                 concat_dual=True,
                 negative_slope_primal=0.2,
                 negative_slope_dual=0.2,
                 dropout_primal=0,
                 dropout_dual=0,
                 bias_primal=False,
                 bias_dual=False,
                 add_self_loops_to_dual_graph=False,
                 allow_pooling_consecutive_edges=True,
                 aggr_primal_features_pooling='mean',
                 aggr_dual_features_pooling='mean',
                 use_decreasing_attention_coefficients=True,
                 return_node_to_cluster=False,
                 log_ratios_new_old_primal_nodes=True,
                 log_ratios_new_old_primal_edges=False):
        super(DualPrimalMeshClassifier, self).__init__()
        # Checks on arguments.
        assert (norm_layer_type in [None, 'group_norm', 'batch_norm'])
        assert (len(conv_primal_out_res) == len(conv_dual_out_res))
        assert (len([
            arg for arg in [
                num_primal_edges_to_keep, fractions_primal_edges_to_keep,
                primal_attention_coeffs_thresholds
            ] if arg is not None
        ]) <= 1), ("Only one of the arguments `num_primal_edges_to_keep`, "
                   "`fractions_primal_edges_to_keep` and "
                   "`primal_attention_coeffs_thresholds` can be non-None.")
        if (num_primal_edges_to_keep is not None):
            assert (
                len(conv_primal_out_res) == len(num_primal_edges_to_keep)
            ), ("Please specify one number of primal edges to keep for each "
                "pooling layer, each of which follows a convolutional layer. "
                "Set the i-th element of num_primal_edges_to_keep to None if "
                "you wish to perform no pooling after the i-th convolutional "
                "layer.")
        if (fractions_primal_edges_to_keep is not None):
            assert (
                len(conv_primal_out_res) == len(fractions_primal_edges_to_keep)
            ), ("Please specify one fraction of primal edges to keep for each "
                "pooling layer, each of which follows a convolutional layer. "
                "Set the i-th element of fractions_primal_edges_to_keep to "
                "None if you wish to perform no pooling after the i-th "
                "convolutional layer.")
        if (primal_attention_coeffs_thresholds is not None):
            assert (len(conv_primal_out_res) == len(
                primal_attention_coeffs_thresholds
            )), ("Please specify one primal-attention threshold for each "
                 "pooling layer, each of which follows a convolutional layer. "
                 "Set the i-th element of primal_attention_coeffs_thresholds "
                 "to None if you wish to perform no pooling after the i-th "
                 "convolutional layer.")

        if (single_dual_nodes):
            assert (undirected_dual_edges), (
                "The dual-graph configuration with single dual nodes and "
                "directed dual edges is not valid. Please specify a different "
                "configuration.")
        # Save input parameters.
        self.__input_parameters = {
            k: v for k, v in locals().items() if (k[0] != '_' and k != 'self')
        }

        conv_primal_res = [in_channels_primal] + conv_primal_out_res
        conv_dual_res = [in_channels_dual] + conv_dual_out_res

        self.__num_conv_layers = len(conv_primal_res) - 1
        self.__res_last_conv_primal_layer = conv_primal_res[-1]
        self.__res_last_conv_dual_layer = conv_dual_res[-1]
        self.__norm_layer_type = norm_layer_type
        self.__return_node_to_cluster = return_node_to_cluster

        at_least_one_pooling_layer = False
        self._indices_pooling_layers = []
        if (num_primal_edges_to_keep is not None):
            self._indices_pooling_layers = [
                layer_idx
                for layer_idx, coeff in enumerate(num_primal_edges_to_keep)
                if coeff != None
            ]
            at_least_one_pooling_layer |= (len(self._indices_pooling_layers) >
                                           0)
        elif (fractions_primal_edges_to_keep is not None):
            self._indices_pooling_layers = [
                layer_idx for layer_idx, coeff in enumerate(
                    fractions_primal_edges_to_keep) if coeff != None
            ]
            at_least_one_pooling_layer |= (len(self._indices_pooling_layers) >
                                           0)
        elif (primal_attention_coeffs_thresholds is not None):
            self._indices_pooling_layers = [
                layer_idx for layer_idx, coeff in enumerate(
                    primal_attention_coeffs_thresholds) if coeff != None
            ]
            at_least_one_pooling_layer |= (len(self._indices_pooling_layers) >
                                           0)

        if (at_least_one_pooling_layer):
            self.__log_ratios_new_old_primal_nodes = (
                log_ratios_new_old_primal_nodes)
            self.__log_ratios_new_old_primal_edges = (
                log_ratios_new_old_primal_edges)
        else:
            print("Note: will not return ratios between number of new and old "
                  "primal nodes/edges: nor amount of primal edges to pool nor "
                  "primal-attention thresholds were provided, hence no pooling "
                  "layer will be instantiated.")
            self.__log_ratios_new_old_primal_nodes = False
            self.__log_ratios_new_old_primal_edges = False

        self.__use_pooling = at_least_one_pooling_layer

        # Initialize network.
        for idx_conv_layer in range(self.__num_conv_layers):
            # - For all layers but the first one, if multi-head attention with
            #   concatenation is used, then the number of input channels needs
            #   to take number of attention heads into account.
            out_channels_primal_prev_layer = conv_primal_res[idx_conv_layer]
            if (concat_primal and idx_conv_layer > 0):
                out_channels_primal_prev_layer *= heads
            out_channels_dual_prev_layer = conv_dual_res[idx_conv_layer]
            if (concat_dual and idx_conv_layer > 0):
                out_channels_dual_prev_layer *= heads
            setattr(
                self, f'block{idx_conv_layer}',
                DualPrimalResDownConv(
                    in_channels_primal=out_channels_primal_prev_layer,
                    in_channels_dual=out_channels_dual_prev_layer,
                    out_channels_primal=conv_primal_res[idx_conv_layer + 1],
                    out_channels_dual=conv_dual_res[idx_conv_layer + 1],
                    norm_layer_type=norm_layer_type,
                    num_groups_norm_layer=num_groups_norm_layer,
                    single_dual_nodes=single_dual_nodes,
                    undirected_dual_edges=undirected_dual_edges,
                    num_primal_edges_to_keep=(
                        num_primal_edges_to_keep[idx_conv_layer]
                        if num_primal_edges_to_keep is not None else None),
                    fraction_primal_edges_to_keep=(
                        fractions_primal_edges_to_keep[idx_conv_layer] if
                        fractions_primal_edges_to_keep is not None else None),
                    primal_attention_coeffs_threshold=(
                        primal_attention_coeffs_thresholds[idx_conv_layer]
                        if primal_attention_coeffs_thresholds is not None else
                        None),
                    num_res_blocks=num_res_blocks,
                    heads=heads,
                    concat_primal=concat_primal,
                    concat_dual=concat_dual,
                    negative_slope_primal=negative_slope_primal,
                    negative_slope_dual=negative_slope_dual,
                    dropout_primal=dropout_primal,
                    dropout_dual=dropout_dual,
                    bias_primal=bias_primal,
                    bias_dual=bias_dual,
                    add_self_loops_to_dual_graph=add_self_loops_to_dual_graph,
                    allow_pooling_consecutive_edges=
                    allow_pooling_consecutive_edges,
                    aggr_primal_features_pooling='mean',
                    aggr_dual_features_pooling='mean',
                    use_decreasing_attention_coefficients=(
                        use_decreasing_attention_coefficients),
                    log_ratio_new_old_primal_nodes=
                    log_ratios_new_old_primal_nodes,
                    log_ratio_new_old_primal_edges=
                    log_ratios_new_old_primal_edges))

        # Fully-connected layers.
        res_previous_layer = self.__res_last_conv_dual_layer

        if (concat_primal):
            res_previous_layer *= heads
        self.fc1 = Linear(in_features=res_previous_layer,
                          out_features=num_output_units_fc)
        self.fc2 = Linear(in_features=num_output_units_fc,
                          out_features=num_classes)

    @property
    def input_parameters(self):
        return self.__input_parameters

    def forward(self, primal_graph_batch, dual_graph_batch,
                primal_edge_to_dual_node_idx_batch):
        r"""Forward pass. Retrieves the global descriptor for each shape in the
            batch.

        Args:
            primal_graph_batch (torch.data.batch.Batch): Data structure
                containing the primal graphs in the batch.
            dual_graph_batch (torch.data.batch.Batch): Data structure containing
                the dual graphs in the batch.
            primal_edge_to_dual_node_idx_batch (dict): Dictionary representing
                the associations between primal-graph edges (tuple) and
                dual-graph nodes for all the graphs in the batch, as a single
                dictionary.

        Returns:
            out (torch.Tensor of shape `[batch_size, num_classes]`): Per-class
                output scores associated to each shape in the batch.
            log_info (pd_mesh_net.nn.pool.PoolingLogMultipleLayers): Logging
                information. If at least one of the class arguments
                `return_node_to_cluster` and `log_ratios_new_old_primal_nodes`
                is True, contains the following attributes:
                - node_to_cluster (dict of list, optional): If the class
                    argument `return_node_to_cluster` is True, dict of
                    `num_pooling_layers` lists, where `num_pooling_layers` is
                    the number of pooling layers in the network. The i-th list
                    is of length `num_nodes_i` - where `num_nodes_i` is the
                    number of nodes in the primal graph before the i-th pooling
                    operation. The j-th element of the list with key i will
                    contain the index of the face cluster to which the j-th face
                    in the mesh inputted to the i-th pooling layer belongs after
                    the i-th pooling operation. If no pooling layer is present
                    in the network, None. If the class argument
                    `return_node_to_cluster` is False, the attribute is not
                    returned.
                - ratios_new_old_primal_nodes (dict of torch.Tensor): If the
                    class argument `log_ratios_new_old_primal_nodes` is True,
                    dict of `num_pooling_layers` tensors, where
                    `num_pooling_layers` is the number of pooling layers in the
                    network. Each of the `num_pooling_layers` tensors is of
                    length `num_samples` - where `num_samples` is the number of
                    samples in the batch. If the j-th pooling layer is not
                    inserted in the network, the entry with key j will be None.
                    Otherwise, the i-th element of the tensor with key j will
                    contain the ratio between the number of primal nodes in the
                    new, pooled graph and the number of primal nodes before
                    pooling, for the i-th sample in the batch and with pooling
                    applied by the pooling layer that follows the j-th
                    convolutional layer in the network. If the class argument
                    `log_ratios_new_old_primal_nodes` is False, None.
                - ratios_new_old_primal_edges (dict of torch.Tensor): If the
                    class argument `log_ratios_new_old_primal_edges` is True,
                    dict of `num_pooling_layers` tensors, where
                    `num_pooling_layers` is the number of pooling layers in the
                    network. Each of the `num_pooling_layers` tensors is of
                    length `num_samples` - where `num_samples` is the number of
                    samples in the batch. If the j-th pooling layer is not
                    inserted in the network, the entry with key j will be None.
                    Otherwise, the i-th element of the tensor with key j will
                    contain the ratio between the number of primal edges in the
                    new, pooled graph and the number of primal edges before
                    pooling, for the i-th sample in the batch and with pooling
                    applied by the pooling layer that follows the j-th
                    convolutional layer in the network. If the class argument
                    `log_ratios_new_old_primal_nodes` is False, None.
        """
        if (self.__log_ratios_new_old_primal_nodes):
            ratios_new_old_primal_nodes_per_pooling_layer = {
                pooling_layer_idx: None
                for pooling_layer_idx in self._indices_pooling_layers
            }
            ratios_new_old_primal_edges_per_pooling_layer = {
                pooling_layer_idx: None
                for pooling_layer_idx in self._indices_pooling_layers
            }
        else:
            ratios_new_old_primal_nodes_per_pooling_layer = None
            ratios_new_old_primal_edges_per_pooling_layer = None
        node_to_cluster = None
        # Forward pass through the first part of the network (chain of
        # dual-primal mesh-convolution layer with skip connections,
        # normalization layers.
        for block_idx in range(self.__num_conv_layers):
            (primal_graph_batch, dual_graph_batch,
             primal_edge_to_dual_node_idx_batch, log_info, _, _) = getattr(
                 self,
                 f'block{block_idx}')(primal_graph_batch=primal_graph_batch,
                                      dual_graph_batch=dual_graph_batch,
                                      primal_edge_to_dual_node_idx_batch=
                                      primal_edge_to_dual_node_idx_batch)

            if (self.__return_node_to_cluster):
                # Store the correspondence between original nodes and new
                # nodes after pooling.
                if (node_to_cluster is None):
                    node_to_cluster = {}
                node_to_cluster[block_idx] = (
                    log_info.old_primal_node_to_new_one.tolist())
            if (self.__log_ratios_new_old_primal_edges and
                    log_info.ratio_new_old_primal_edges is not None):
                ratios_new_old_primal_edges_per_pooling_layer[
                    block_idx] = log_info.ratio_new_old_primal_edges
            if (self.__log_ratios_new_old_primal_nodes and
                    log_info.ratio_new_old_primal_nodes is not None):
                ratios_new_old_primal_nodes_per_pooling_layer[
                    block_idx] = log_info.ratio_new_old_primal_nodes

        x_primal, batch_primal = primal_graph_batch.x, primal_graph_batch.batch
        x_dual, batch_dual = dual_graph_batch.x, dual_graph_batch.batch

        # Apply (average) pooling operation on the output of the first part of
        # the network, before the fully-connected layers.
        # Before: shape = (#total_num_primal_nodes_in_batch,
        #                  #channels_last_convolution).
        x_input_fc = avg_pool_x(cluster=batch_dual, x=x_dual,
                                batch=batch_dual)[0]
        # After: shape = (batch_size, #channels_last_convolution).

        # The output of the pooling layer is passed through a first
        # fully-connected layer, with ReLU activation.
        x = F.relu(self.fc1(x_input_fc))
        # The output of the first fully-connected layer is passed through a
        # second fully-connected layer, yielding the final scores.
        out = self.fc2(x)

        #NOTE: The output features represent unnormalized per-class scores and
        # should therefore be soft-maxed and, e.g., undergo negative log (one
        # can combine the two operations with cross-entropy loss).

        log_info = PoolingLogMultipleLayers(
            ratios_new_old_primal_nodes=
            ratios_new_old_primal_nodes_per_pooling_layer,
            ratios_new_old_primal_edges=
            ratios_new_old_primal_edges_per_pooling_layer,
            node_to_cluster=node_to_cluster)

        return out, log_info
