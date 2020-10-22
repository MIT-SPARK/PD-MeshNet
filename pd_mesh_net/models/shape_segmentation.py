import torch
import torch.nn.functional as F
from torch_geometric.nn.norm import BatchNorm
from torch_scatter import scatter_mean

from pd_mesh_net.models import (BaseDualPrimalModel, DualPrimalDownConv,
                                DualPrimalResDownConv, DualPrimalUpConv)
from pd_mesh_net.nn import (GATConvNoSelfLoops, PoolingLogMultipleLayers,
                            DualPrimalConv)


class DualPrimalMeshSegmenter(BaseDualPrimalModel):
    r"""Architecture for learning a class labels associated to faces in meshes
    (segmentation).
    Similar to an analogous network from MeshCNN
    (https://github.com/ranahanocka/MeshCNN/).

    Args:
        in_channels_primal, in_channels_dual (int): Size of each input sample
            from the primal/dual graph respectively.
        conv_primal_out_res, conv_dual_out_res (list of int): Number of output
            channels of the primal/dual layers respectively in each
            `DualPrimalDownConv`/`DualPrimalResDownConv` block, i.e., dimension
            of the feature vectors outputted by the primal/dual layers
            respectively of these blocks. Note: a further block is added at the
            end of the network, having number of output primal channels equal to
            the number of classes in the dataset (cf. argument `num_classes`)
            and number of output dual channels equal to the number of output
            dual channels of the previous `DualPrimalDownConv`/
            `DualPrimalResDownConv` block; no pooling is performed in this last
            block.
        num_classes (int): Number of classes in the dataset.
        single_dual_nodes (bool): If True, it will be assumed that the dual
            graphs have single nodes; otherwise, it will be assumed that they
            have double nodes. Cf. :obj:`pd_mesh_net.utils.GraphCreator`.
        undirected_dual_edges (bool): If True, it will be assumed that every
            directed edge in the dual graphs has an opposite directed edge;
            otherwise, it will be assumed that directed edges in the dual graphs
            do not have an opposite directed edge.
            Cf. :obj:`pd_mesh_net.utils.GraphCreator`.
        use_dual_primal_res_down_conv_blocks (bool, optional): If True,
            `DualPrimalResDownConv` blocks are used instead of the
            `DualPrimalDownConv` blocks. (default: :obj:`False`)
        norm_layer_type (str or None, optional): Type of normalization layer to
            be used after each convolutional layer, both primal and dual.
            Possible options are: `None` (no normalization layer),
            `'group_norm'` (group normalization), `'batch_norm'` (batch
            normalization). Only used if `DualPrimalResDownConv` blocks are used
            (cf. argument `use_dual_primal_res_down_conv_blocks`).
            (default: :obj:`'group_norm'`)
        num_groups_norm_layer (int, optional): If `norm_layer_type` is
            'group_norm', number of groups in each normalization layer. Only
            used if `DualPrimalResDownConv` blocks are used (cf. argument
            `use_dual_primal_res_down_conv_blocks`). (default: :obj:`16`)
        num_primal_edges_to_keep (list of int/None, optional):
            Either None or a list of int values.
            - If not None, the value of the i-th element represents the target
              number of primal edges to keep in the pooling layer after in the
              i-th `DualPrimalDownConv`/`DualPrimalResDownConv` block;
            For each valid i, at most one of `num_primal_edges_to_keep[i]`
            `fractions_primal_edges_to_keep[i]` and
            `primal_attention_coeffs_thresholds[i]` can be not None. If
            `num_primal_edges_to_keep`, `fractions_primal_edges_to_keep` and
            `primal_attention_coeffs_thresholds` are all None, no pooling layers
            are inserted in the network. (default: :obj:`None`)
        fractions_primal_edges_to_keep (list of float/None, optional):
            Either None or a list of float values.
            - If not None, the value of the i-th element represents the target
              fraction of primal edges to keep in the pooling layer after in the
              i-th `DualPrimalDownConv`/`DualPrimalResDownConv` block;
            For each valid i, at most one of `num_primal_edges_to_keep[i]`
            `fractions_primal_edges_to_keep[i]` and
            `primal_attention_coeffs_thresholds[i]` can be not None. If
            `num_primal_edges_to_keep`, `fractions_primal_edges_to_keep` and
            `primal_attention_coeffs_thresholds` are all None, no pooling layers
            are inserted in the network. (default: :obj:`None`)
        primal_attention_coeffs_thresholds (list of float/None, optional):
            Either None or a list of float values.
            - If not None, the value of the i-th element represents the
              threshold used in the pooling layer in the i-th
              `DualPrimalDownConv`/`DualPrimalResDownConv` block to pool primal
              edges for which the attention coefficient is above/below this
              value, depending on the argument
              `use_decreasing_attention_coefficients`;
            For each valid i, at most one of `num_primal_edges_to_keep[i]`
            `fractions_primal_edges_to_keep[i]` and
            `primal_attention_coeffs_thresholds[i]` can be not None. If
            `num_primal_edges_to_keep`, `fractions_primal_edges_to_keep` and
            `primal_attention_coeffs_thresholds` are all None, no pooling layers
            are inserted in the network. (default: :obj:`None`)
        num_res_blocks (int, optional): Number of residual blocks per each
            `DualPrimalDownConv` block (cf. `num_skips` in
            `DualPrimalDownConv`). (default: :obj:`3`).
        heads (int, optional): Number of attention heads associated to each of
            the primal and dual convolutional layers in the
            `DualPrimalDownConv`/`DualPrimalResDownConv` blocks. Note: the
            additional `DualPrimalDownConv`/`DualPrimalResDownConv` block (cf.
            arguments `conv_primal_out_res` and `conv_dual_out_res`) has always
            a single attention head. (default: :obj:`1`)
        concat_primal, concat_dual (bool, optional): If set to :obj:`False`, the
            attention heads associated respectively to the primal and to the
            dual convolutional layers in the `DualPrimalDownConv`/
            `DualPrimalResDownConv` blocks are averaged instead of concatenated.
            Note: in the additional `DualPrimalDownConv`/`DualPrimalResDownConv`
            block (cf. arguments `conv_primal_out_res` and `conv_dual_out_res`)
            attention heads are always averaged. (default: :obj:`True`)
        negative_slope_primal, negative_slope_dual (float, optional): LeakyReLU
            angle of the negative slope, respectively for the layers associated
            to the primal graph and for the layers associated to the dual graph.
            (default: :obj:`0.2`)
        dropout_primal, dropout_dual (float, optional): Dropout probability of
            the normalized attention coefficients which exposes each node,
            respectively of the primal and of the dual convolutional layers in
            the `DualPrimalDownConv`/`DualPrimalResDownConv` blocks, to a
            stochastically sampled neighborhood during training.
            (default: :obj:`0`)
        bias_primal, bias_dual (bool, optional): If set to :obj:`False`, the
            layers associated respectively to the primal and to the dual graph
            will not learn an additive bias. (default: :obj:`False`)
        add_self_loops_to_dual_graph (bool, optional): If set to :obj:`True`,
            regular graph-attention convolutional layers are instantiated as
            dual convolutional layers in the `DualPrimalDownConv`/
            `DualPrimalResDownConv` blocks, thus self-loops are added to the
            dual graph. Furthermore, self-loops are added to dual graphs
            generated by the pooling layers in the `DualPrimalDownConv`/
            `DualPrimalResDownConv` blocks. If set to :obj:`False` instead, a
            modified version of the graph-attention convolutional layer is
            instantiated for the dual convolutional layers in the
            `DualPrimalDownConv`/`DualPrimalResDownConv` blocks, with no
            addition of self-loops to the latter, and no self loops are added to
            the dual graphs generated by the pooling layers in the
            `DualPrimalDownConv`/`DualPrimalResDownConv` blocks.
            (default: :obj:`False`)
        allow_pooling_consecutive_edges (bool, optional): If True, no
            restrictions are put on the primal edges that can be pooled by the
            pooling layers in the model. If False, a primal edge can only be
            pooled if no primal nodes to which it belongs have been pooled
            previously. Setting this argument to False is only compatible with
            top-K pooling (cf. arguments `num_primal_edges_to_keep` and
            `fraction_primal_edges_to_keep`). (default: :obj:`True`)
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
        gat_in_last_block (bool, optional): If True, the additional block at the
            end of the model (cf. arguments `conv_primal_out_res` and
            `conv_dual_out_res`) is a simple GATConvNoSelfLoops instantiated
            from the previous primal layer. Otherwise, it is a regular
            `DualPrimalDownConv`/`DualPrimalResDownConv` block (cf. arguments
            `conv_primal_out_res`, `conv_dual_out_res`, `heads`,
            `concat_primal`, `concat_dual` and attribute `final_block` for more
            details on the parameters). (default: :obj:`False`)
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
            layer in the j-th `DualPrimalDownConv`/`DualPrimalResDownConv` block
            in the network. (default: :obj:`True`)
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
            layer in the j-th `DualPrimalDownConv`/`DualPrimalResDownConv` block
            in the network. (default: :obj:`False`)
        return_info_for_unpooling (bool, optional): If True, each call to the
            forward method will also return:
            - Two lists with `torch_geometric.data.batch.Batch` structures
              representing the graphs before the pooling operation is performed,
              for each block that contains a pooling layer;
            - A list containing the pooling logs for each block that contains a
              pooling layer (cf. `pd_mesh_net.nn.pool.PoolingInfo).
            This may be required when using the network in an encoder-decoder
            architecture. If False, the three lists will contain None values.
            (default: :obj:`False`)
        do_not_add_final_block (bool, optional): If True, the additional final
            block (cf., e.g., attribute `final_block`) is not added. This can
            be useful to spare memory in case the block is not needed (e.g.,
            when using the model as part of an encoder-decoder structure, and
            only the `forward_encoder_only` method is used).
            (default: :obj:`False`)

    Attributes:
        final_block (pd_mesh_net.models.down_conv.DualPrimalDownConv or
            pd_mesh_net.models.down_conv.DualPrimalResDownConv or
            pd_mesh_net.nn.conv.GATConvNoSelfLoops): Final block, either a
            down-convolution/down-convolution resiudal block or a
            GATConvNoSelfLoops on the primal graph, depending on the input
            argument `gat_in_last_block`. In both cases, the number of output
            primal channels is equal to the number of classes in the dataset
            (cf. argument `num_classes`). If using a down-convolution/
            down-convolution residual block, the number of output dual channels
            is equal to the number of output dual channels of the previous
            `DualPrimalDownConv`/`DualPrimalResDownConv` block, and no pooling
            is performed in this last block.
        _indices_pooling_layers (list): If the element `i` is in this list, then
            the i-th `DualPrimalDownConv`/`DualPrimalResDownConv` block contains
            a pooling layer.
        ---
        block{i} (pd_mesh_net.models.down_conv.DualPrimalDownConv or
            pd_mesh_net.models.down_conv.DualPrimalResDownConv), `i` in
            `{0, ..., self.__num_down_conv_blocks - 1}`: `i`-th down-convolution
            or dual-primal residual down-convolutional block, depending on the
            input argument `use_dual_primal_res_conv_blocks`.
    """

    def __init__(self,
                 in_channels_primal,
                 in_channels_dual,
                 conv_primal_out_res,
                 conv_dual_out_res,
                 num_classes,
                 single_dual_nodes,
                 undirected_dual_edges,
                 use_dual_primal_res_down_conv_blocks=False,
                 norm_layer_type='group_norm',
                 num_groups_norm_layer=16,
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
                 gat_in_last_block=False,
                 return_node_to_cluster=False,
                 log_ratios_new_old_primal_nodes=True,
                 log_ratios_new_old_primal_edges=False,
                 return_info_for_unpooling=False,
                 do_not_add_final_block=False):

        super(DualPrimalMeshSegmenter, self).__init__()
        # Checks on arguments.
        assert (len(conv_primal_out_res) == len(conv_dual_out_res))

        _nonnone_args = []
        for _arg in [
                num_primal_edges_to_keep, fractions_primal_edges_to_keep,
                primal_attention_coeffs_thresholds
        ]:
            if (_arg is not None):
                assert (
                    isinstance(_arg, list) and
                    len(_arg) == len(conv_primal_out_res)
                ), ("The arguments `num_primal_edges_to_keep`, "
                    "`fractions_primal_edges_to_keep` and "
                    "`primal_attention_coeffs_thresholds` must be either None "
                    "or of type list. In the latter case the list must contain "
                    "one item (optionally `None`) for each down-convolutional "
                    "block.")
                _nonnone_args.append(_arg)
        _at_least_one_pooling_layer = False
        self._indices_pooling_layers = []
        for _block_idx in range(len(conv_primal_out_res)):
            _num_nonnone_values = 0
            for _nonnone_arg in _nonnone_args:
                if (_nonnone_arg[_block_idx] is not None):
                    _num_nonnone_values += 1
            assert (_num_nonnone_values <= 1), (
                "For each down-convolutional block at most one of the "
                "associated values in `num_primal_edges_to_keep`, "
                "`fractions_primal_edges_to_keep` and "
                "`primal_attention_coeffs_thresholds` can be non-None.")
            if (_num_nonnone_values > 0):
                self._indices_pooling_layers.append(_block_idx)
                _at_least_one_pooling_layer = True

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

        self.__num_down_conv_blocks = len(conv_primal_res) - 1
        self.__use_dual_primal_res_down_conv_blocks = (
            use_dual_primal_res_down_conv_blocks)
        self.__res_last_conv_primal_layer = conv_primal_res[-1]
        self.__res_last_conv_dual_layer = conv_dual_res[-1]
        self.__gat_in_last_block = gat_in_last_block
        self.__return_node_to_cluster = return_node_to_cluster
        self.__return_info_for_unpooling = return_info_for_unpooling
        self.__do_not_add_final_block = do_not_add_final_block

        if (_at_least_one_pooling_layer):
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

        # Initialize network.
        for block_idx in range(self.__num_down_conv_blocks):
            # Stack DualPrimalDownConv/DualPrimalResDownConv blocks.
            # - For all layers but the first one, if multi-head attention with
            #   concatenation is used, then the number of input channels needs
            #   to take number of attention heads into account.
            out_channels_primal_prev_layer = conv_primal_res[block_idx]
            if (concat_primal and block_idx > 0):
                out_channels_primal_prev_layer *= heads
            out_channels_dual_prev_layer = conv_dual_res[block_idx]
            if (concat_dual and block_idx > 0):
                out_channels_dual_prev_layer *= heads
            # - If the block contains a pooling layer, and if required, store
            #   the graphs as they are before applying the pooling operation.
            should_return_info_for_unpooling_in_layer = (
                return_info_for_unpooling and
                block_idx in self._indices_pooling_layers)
            if (self.__use_dual_primal_res_down_conv_blocks):
                # - DualPrimalResDownConv block.
                setattr(
                    self, f'block{block_idx}',
                    DualPrimalResDownConv(
                        in_channels_primal=out_channels_primal_prev_layer,
                        in_channels_dual=out_channels_dual_prev_layer,
                        out_channels_primal=conv_primal_res[block_idx + 1],
                        out_channels_dual=conv_dual_res[block_idx + 1],
                        norm_layer_type=norm_layer_type,
                        num_groups_norm_layer=num_groups_norm_layer,
                        single_dual_nodes=single_dual_nodes,
                        undirected_dual_edges=undirected_dual_edges,
                        num_primal_edges_to_keep=num_primal_edges_to_keep[
                            block_idx]
                        if num_primal_edges_to_keep is not None else None,
                        fraction_primal_edges_to_keep=(
                            fractions_primal_edges_to_keep[block_idx]
                            if fractions_primal_edges_to_keep is not None else
                            None),
                        primal_attention_coeffs_threshold=(
                            primal_attention_coeffs_thresholds[block_idx]
                            if primal_attention_coeffs_thresholds is not None
                            else None),
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
                        add_self_loops_to_dual_graph=
                        add_self_loops_to_dual_graph,
                        allow_pooling_consecutive_edges=(
                            allow_pooling_consecutive_edges),
                        aggr_primal_features_pooling=(
                            aggr_primal_features_pooling),
                        aggr_dual_features_pooling=aggr_dual_features_pooling,
                        use_decreasing_attention_coefficients=(
                            use_decreasing_attention_coefficients),
                        log_ratio_new_old_primal_nodes=(
                            log_ratios_new_old_primal_nodes),
                        log_ratio_new_old_primal_edges=(
                            log_ratios_new_old_primal_edges),
                        return_old_dual_node_to_new_dual_node=(
                            should_return_info_for_unpooling_in_layer),
                        return_graphs_before_pooling=(
                            should_return_info_for_unpooling_in_layer)))
            else:
                # - DualPrimalDownConv block.
                setattr(
                    self, f'block{block_idx}',
                    DualPrimalDownConv(
                        in_channels_primal=out_channels_primal_prev_layer,
                        in_channels_dual=out_channels_dual_prev_layer,
                        out_channels_primal=conv_primal_res[block_idx + 1],
                        out_channels_dual=conv_dual_res[block_idx + 1],
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
                        add_self_loops_to_dual_graph=
                        add_self_loops_to_dual_graph,
                        num_skips=num_res_blocks,
                        num_primal_edges_to_keep=num_primal_edges_to_keep[
                            block_idx]
                        if num_primal_edges_to_keep is not None else None,
                        fraction_primal_edges_to_keep=(
                            fractions_primal_edges_to_keep[block_idx]
                            if fractions_primal_edges_to_keep is not None else
                            None),
                        primal_attention_coeff_threshold=(
                            primal_attention_coeffs_thresholds[block_idx]
                            if primal_attention_coeffs_thresholds is not None
                            else None),
                        allow_pooling_consecutive_edges=(
                            allow_pooling_consecutive_edges),
                        use_decreasing_attention_coefficients=(
                            use_decreasing_attention_coefficients),
                        log_ratio_new_old_primal_nodes=(
                            log_ratios_new_old_primal_nodes),
                        log_ratio_new_old_primal_edges=(
                            log_ratios_new_old_primal_edges),
                        return_old_dual_node_to_new_dual_node=(
                            should_return_info_for_unpooling_in_layer),
                        return_graphs_before_pooling=(
                            should_return_info_for_unpooling_in_layer)))

        if (not do_not_add_final_block):
            # Add a final DualPrimalDownConv/DualPrimalResDownConv block without
            # pooling/GATConvNoSelfLoops.
            out_channels_primal_prev_layer = conv_primal_res[-1]
            if (concat_primal):
                out_channels_primal_prev_layer *= heads
            out_channels_dual_prev_layer = conv_dual_res[-1]
            if (concat_dual):
                out_channels_dual_prev_layer *= heads
            if (gat_in_last_block):
                self.final_block = GATConvNoSelfLoops(
                    in_channels=out_channels_primal_prev_layer,
                    out_channels=num_classes,
                    heads=1,
                    concat=False,
                    negative_slope=negative_slope_primal,
                    dropout=dropout_primal,
                    bias=bias_primal)
            else:
                if (self.__use_dual_primal_res_down_conv_blocks):
                    self.final_block = DualPrimalResDownConv(
                        in_channels_primal=out_channels_primal_prev_layer,
                        in_channels_dual=out_channels_dual_prev_layer,
                        out_channels_primal=num_classes,
                        out_channels_dual=conv_dual_res[-1],
                        norm_layer_type=None,
                        num_groups_norm_layer=num_groups_norm_layer,
                        single_dual_nodes=single_dual_nodes,
                        undirected_dual_edges=undirected_dual_edges,
                        num_res_blocks=num_res_blocks,
                        heads=1,
                        concat_primal=False,
                        concat_dual=False,
                        negative_slope_primal=negative_slope_primal,
                        negative_slope_dual=negative_slope_dual,
                        dropout_primal=dropout_primal,
                        dropout_dual=dropout_dual,
                        bias_primal=bias_primal,
                        bias_dual=bias_dual,
                        add_self_loops_to_dual_graph=(
                            add_self_loops_to_dual_graph))
                else:
                    self.final_block = DualPrimalDownConv(
                        in_channels_primal=out_channels_primal_prev_layer,
                        in_channels_dual=out_channels_dual_prev_layer,
                        out_channels_primal=num_classes,
                        out_channels_dual=conv_dual_res[-1],
                        heads=1,
                        concat_primal=False,
                        concat_dual=False,
                        negative_slope_primal=negative_slope_primal,
                        negative_slope_dual=negative_slope_dual,
                        dropout_primal=dropout_primal,
                        dropout_dual=dropout_dual,
                        bias_primal=bias_primal,
                        bias_dual=bias_dual,
                        single_dual_nodes=single_dual_nodes,
                        undirected_dual_edges=undirected_dual_edges,
                        add_self_loops_to_dual_graph=
                        add_self_loops_to_dual_graph,
                        num_skips=num_res_blocks)

    @property
    def input_parameters(self):
        return self.__input_parameters

    def forward(self, primal_graph_batch, dual_graph_batch,
                primal_edge_to_dual_node_idx_batch):
        r"""Forward pass. Retrieves unnormalized per-class scores for each face
        in the input mesh (i.e., for each node in the input primal graph).

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
            out (list of torch.Tensor of shape
                `[num_primal_nodes_batch, num_classes]`, where
                `num_primal_nodes_batch` is the number of primal nodes in the
                input batch): The i-th tensor in the list contains the per-class
                unnormalized scores associated to the i-th primal node in the
                input batch.
            log_info (namedtuple): Logging information. If at least one of the
                class arguments `return_node_to_cluster` and
                `log_ratios_new_old_primal_nodes` is True, contains the
                following attributes:
                - node_to_cluster (dict of torch.Tensor, optional): If the class
                    argument `return_node_to_cluster` is True, dict of
                    `num_pooling_layers` tensors, where `num_pooling_layers` is
                    the number of pooling layers in the network. Cf. method
                    `forward_encoder_only` for more details.
                - ratios_new_old_primal_nodes (dict of torch.Tensor, optional):
                    If the class argument `log_ratios_new_old_primal_nodes` is
                    True, dict of `num_pooling_layers` tensors, where
                    `num_pooling_layers` is the number of pooling layers in the
                    network. Cf. method `forward_encoder_only` for more details.
                - ratios_new_old_primal_edges (dict of torch.Tensor, optional):
                    If the class argument `log_ratios_new_old_primal_edges` is
                    True, dict of `num_pooling_layers` tensors, where
                    `num_pooling_layers` is the number of pooling layers in the
                    network. Cf. method `forward_encoder_only` for more details.
        """
        assert (not self.__do_not_add_final_block), (
            "The additional convolution layer at the end of the network should "
            "be added if the network is not used as encoder only. Please set "
            "the class argument `do_not_add_final_block` to False.")
        # Forward pass through the 'encoder'.
        (primal_graph_batch, dual_graph_batch,
         primal_edge_to_dual_node_idx_batch, node_to_cluster,
         ratios_new_old_primal_nodes_per_down_conv_block,
         ratios_new_old_primal_edges_per_down_conv_block, _, _,
         _) = self.forward_encoder_only(primal_graph_batch=primal_graph_batch,
                                        dual_graph_batch=dual_graph_batch,
                                        primal_edge_to_dual_node_idx_batch=(
                                            primal_edge_to_dual_node_idx_batch))
        # Forward pass through the final block.
        if (self.__gat_in_last_block):
            primal_graph_x = self.final_block(
                x=primal_graph_batch.x,
                edge_index=primal_graph_batch.edge_index)
            primal_graph_batch.x = primal_graph_x
        else:
            (primal_graph_batch, dual_graph_batch,
             primal_edge_to_dual_node_idx_batch, log_info, _,
             _) = self.final_block(primal_graph_batch=primal_graph_batch,
                                   dual_graph_batch=dual_graph_batch,
                                   primal_edge_to_dual_node_idx_batch=(
                                       primal_edge_to_dual_node_idx_batch))

        if (len(node_to_cluster) == 0):
            # If no pooling layer was inserted, each face corresponds to a
            # different cluster, and thus has its own label already.
            out = primal_graph_batch.x
        else:
            # For each pooling layer, map the output nodes to the input nodes,
            # starting from the last one (i.e., the one closest to the end of
            # the network) and going backwards, until the output nodes of the
            # last layer have been mapped to the input nodes of the first layer.
            last_to_first_pooling_layer_indices = sorted(
                node_to_cluster.keys())[::-1]
            node_to_clusters_next_layer = node_to_cluster[
                last_to_first_pooling_layer_indices[0]]
            for pooling_layer_idx in last_to_first_pooling_layer_indices[1:]:
                node_to_clusters_next_layer = node_to_clusters_next_layer[
                    node_to_cluster[pooling_layer_idx]]
            # Assign to each primal node in the input batch the feature of the
            # corresponding primal node in the graph outputted by the last block.
            out = primal_graph_batch.x[node_to_clusters_next_layer]

        #NOTE: The output features represent unnormalized per-class scores and
        # should therefore be soft-maxed and, e.g., undergo negative log (one
        # can combine the two operations with cross-entropy loss).

        log_info = PoolingLogMultipleLayers(
            ratios_new_old_primal_nodes=
            ratios_new_old_primal_nodes_per_down_conv_block,
            ratios_new_old_primal_edges=
            ratios_new_old_primal_edges_per_down_conv_block,
            node_to_cluster=node_to_cluster
            if self.__return_node_to_cluster else None)

        return out, log_info

    def forward_encoder_only(self, primal_graph_batch, dual_graph_batch,
                             primal_edge_to_dual_node_idx_batch):
        r"""Forward pass of the encoder part of the network only, i.e., of the
        part before the additional final block.

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
            primal_graph_batch (torch.data.batch.Batch): Data structure
                containing the primal graphs in the batch after the forward
                pass.
            dual_graph_batch (torch.data.batch.Batch): Data structure containing
                the dual graphs in the batch after the forward pass.
            primal_edge_to_dual_node_idx_batch (dict): Dictionary representing
                the associations between primal-graph edges (tuple) and
                dual-graph nodes for all the graphs in the batch, as a single
                dictionary, after the forward pass.
            node_to_cluster (dict of torch.Tensor): Dict of `num_pooling_layers`
                tensors, where `num_pooling_layers` is the number of pooling
                layers in the network. The i-th tensor is of length
                `num_nodes_i` - where `num_nodes_i` is the number of nodes in
                the primal graph before the i-th pooling operation. The j-th
                element of the tensor with key i will contain the index of the
                face cluster to which the j-th face in the mesh inputted to the
                i-th pooling layer belongs after the i-th pooling operation.
            ratios_new_old_primal_nodes (dict of torch.Tensor):
                If the class argument `log_ratios_new_old_primal_nodes` is
                True, dict of `num_pooling_layers` tensors, where
                `num_pooling_layers` is the number of pooling layers in the
                network. Each of the `num_pooling_layers` tensors is of
                length `num_samples` - where `num_samples` is the number of
                samples in the batch. The i-th element of the list with key
                j will contain the ratio between the number of primal nodes
                in the new, pooled graph and the number of primal nodes
                before pooling, for the i-th sample in the batch and with
                pooling applied by the pooling layer in the j-th
                `DualPrimalDownConv`/`DualPrimalResDownConv` block in the
                network.
            ratios_new_old_primal_edges (dict of torch.Tensor, optional):
                If the class argument `log_ratios_new_old_primal_edges` is
                True, dict of `num_pooling_layers` tensors, where
                `num_pooling_layers` is the number of pooling layers in the
                network. Each of the `num_pooling_layers` tensors is of
                length `num_samples` - where `num_samples` is the number of
                samples in the batch. The i-th element of the list with key
                j will contain the ratio between the number of primal edges
                in the new, pooled graph and the number of primal edges
                before pooling, for the i-th sample in the batch and with
                pooling applied by the pooling layer in the j-th
                `DualPrimalDownConv`/`DualPrimalResDownConv` block in the
                network.
            primal_graph_batches_before_pooling (list): If the class argument
                `return_info_for_unpooling` is True and the i-th block contains
                a pooling layer, the i-th element of this list contains the
                batch with the primal graphs as they are before applying the
                pooling operation in the i-th `DualPrimalDownConv`/
                `DualPrimalResDownConv` block. Otherwise, the i-th element is
                None.
            dual_graph_batches_before_pooling (list): If the class argument
                `return_info_for_unpooling` is True and the i-th block contains
                a pooling layer, the i-th element of this list contains the
                batch with the dual graphs as they are before applying the
                pooling operation in the i-th `DualPrimalDownConv`/
                `DualPrimalResDownConv` block. Otherwise, the i-th element is
                None.
            pooling_info_per_block (list): If the class argument
                `return_info_for_unpooling` is True and the i-th block contains
                a pooling layer, the i-th element of this list contains the
                pooling information associated to the pooling layer in the i-th
                `DualPrimalDownConv`/`DualPrimalResDownConv` block (cf.
                `pd_mesh_net.nn.pool.PoolingInfo`). Otherwise, the i-th element
                is None.
        """
        if (self.__log_ratios_new_old_primal_edges):
            ratios_new_old_primal_edges_per_down_conv_block = {
                pooling_layer_idx: None
                for pooling_layer_idx in self._indices_pooling_layers
            }
        else:
            ratios_new_old_primal_edges_per_down_conv_block = None
        if (self.__log_ratios_new_old_primal_nodes):
            ratios_new_old_primal_nodes_per_down_conv_block = {
                pooling_layer_idx: None
                for pooling_layer_idx in self._indices_pooling_layers
            }
        else:
            ratios_new_old_primal_nodes_per_down_conv_block = None
        node_to_cluster = {}
        primal_graph_batches_before_pooling = []
        dual_graph_batches_before_pooling = []
        pooling_info_per_block = []
        # Forward pass through the stacked DualPrimalDownConv blocks.
        for block_idx in range(self.__num_down_conv_blocks):
            (primal_graph_batch, dual_graph_batch,
             primal_edge_to_dual_node_idx_batch, log_info,
             primal_graph_batch_before_pooling,
             dual_graph_batch_before_pooling) = getattr(
                 self,
                 f'block{block_idx}')(primal_graph_batch=primal_graph_batch,
                                      dual_graph_batch=dual_graph_batch,
                                      primal_edge_to_dual_node_idx_batch=
                                      primal_edge_to_dual_node_idx_batch)
            if (log_info is not None):
                # Store the correspondence between original nodes and new nodes
                # after pooling.
                if (log_info.old_primal_node_to_new_one is not None):
                    node_to_cluster[
                        block_idx] = log_info.old_primal_node_to_new_one
                # Store ratios between number of primal edges/nodes after and
                # before each pooling operation.
                if (self.__log_ratios_new_old_primal_edges and
                        log_info.ratio_new_old_primal_edges is not None):
                    ratios_new_old_primal_edges_per_down_conv_block[
                        block_idx] = log_info.ratio_new_old_primal_edges
                if (self.__log_ratios_new_old_primal_nodes and
                        log_info.ratio_new_old_primal_nodes is not None):
                    ratios_new_old_primal_nodes_per_down_conv_block[
                        block_idx] = log_info.ratio_new_old_primal_nodes
            if (self.__return_info_for_unpooling):
                pooling_info_per_block.append(log_info)
            else:
                pooling_info_per_block.append(None)

            primal_graph_batches_before_pooling.append(
                primal_graph_batch_before_pooling)
            dual_graph_batches_before_pooling.append(
                dual_graph_batch_before_pooling)

        ratios_new_old_primal_nodes = (
            ratios_new_old_primal_nodes_per_down_conv_block)
        ratios_new_old_primal_edges = (
            ratios_new_old_primal_edges_per_down_conv_block)

        return (primal_graph_batch, dual_graph_batch,
                primal_edge_to_dual_node_idx_batch, node_to_cluster,
                ratios_new_old_primal_nodes, ratios_new_old_primal_edges,
                primal_graph_batches_before_pooling,
                dual_graph_batches_before_pooling, pooling_info_per_block)


class DualPrimalUNetMeshSegmenter(BaseDualPrimalModel):
    r"""Architecture for learning a class labels associated to faces in meshes
    (segmentation), using a UNet-like encoder-decoder architecture. The encoder
    is a `DualPrimalMeshSegmenter` model without additional final block (cf.
    docs of `DualPrimalMeshSegmenter`).
    Based on an analogous network from MeshCNN
    (https://github.com/ranahanocka/MeshCNN/).

    Args:
        in_channels_primal, in_channels_dual (int): Size of each input sample
            from the primal/dual graph respectively.
        conv_primal_out_res, conv_dual_out_res (list of int): All the elements
            up to the second-last of these list represent the number of output
            channels of the primal/dual layers respectively in each
            `DualPrimalDownConv`/`DualPrimalResDownConv` block, i.e., dimension
            of the feature vectors outputted by the primal/dual layers
            respectively of these blocks; for each down-convolutional block, a
            'mirroring' `DualPrimalUpConv` block is created in the decoder part
            of the network that will bring the number of channels back to the
            dimensions of the corresponding down-convolutional block. The last
            element of these list is the number of primal/dual output channels
            of the `DualPrimalConv` layer that is added between the encoder and
            the decoder. A further `DualPrimalConv` layer is finally added at
            the end of the network, after the decoder, having number of output
            primal channels equal to the number of classes in the dataset (cf.
            argument `num_classes`) and number of output dual channels equal to
            the number of output dual channels of the previous layer. Batch
            normalization is also added after the additional layer.
        num_classes (int): Number of classes in the dataset.
        single_dual_nodes (bool): If True, it will be assumed that the dual
            graphs have single nodes; otherwise, it will be assumed that they
            have double nodes. Cf. :obj:`pd_mesh_net.utils.GraphCreator`.
        undirected_dual_edges (bool): If True, it will be assumed that every
            directed edge in the dual graphs has an opposite directed edge;
            otherwise, it will be assumed that directed edges in the dual graphs
            do not have an opposite directed edge.
            Cf. :obj:`pd_mesh_net.utils.GraphCreator`.
        use_dual_primal_res_down_conv_blocks (bool, optional): If True, in the
            encoder `DualPrimalResDownConv` blocks are used instead of the
            `DualPrimalDownConv` blocks. (default: :obj:`False`)
        norm_layer_type (str or None, optional): Type of normalization layer to
            be used after each convolutional layer, both primal and dual.
            Possible options are: `None` (no normalization layer),
            `'group_norm'` (group normalization), `'batch_norm'` (batch
            normalization). Only used if `DualPrimalResDownConv` blocks are used
            (cf. argument `use_dual_primal_res_down_conv_blocks`).
            (default: :obj:`'group_norm'`)
        num_groups_norm_layer (int, optional): If `norm_layer_type` is
            'group_norm', number of groups in each normalization layer in the
            encoder. Only used if `DualPrimalResDownConv` blocks are used (cf.
            argument `use_dual_primal_res_down_conv_blocks`).
            (default: :obj:`16`)
        num_primal_edges_to_keep (list of int/None, optional):
            Either None or a list of int values.
            - If not None, the value of the i-th element represents the target
              number of primal edges to keep in the pooling layer after in the
              i-th `DualPrimalDownConv`/`DualPrimalResDownConv` block;
            For each valid i, at most one of `num_primal_edges_to_keep[i]`
            `fractions_primal_edges_to_keep[i]` and
            `primal_attention_coeffs_thresholds[i]` can be not None. If
            `num_primal_edges_to_keep`, `fractions_primal_edges_to_keep` and
            `primal_attention_coeffs_thresholds` are all None, no pooling layers
            are inserted in the network. (default: :obj:`None`)
        fractions_primal_edges_to_keep (list of float/None, optional):
            Either None or a list of float values.
            - If not None, the value of the i-th element represents the target
              fraction of primal edges to keep in the pooling layer after in the
              i-th `DualPrimalDownConv`/`DualPrimalResDownConv` block;
            For each valid i, at most one of `num_primal_edges_to_keep[i]`
            `fractions_primal_edges_to_keep[i]` and
            `primal_attention_coeffs_thresholds[i]` can be not None. If
            `num_primal_edges_to_keep`, `fractions_primal_edges_to_keep` and
            `primal_attention_coeffs_thresholds` are all None, no pooling layers
            are inserted in the network. (default: :obj:`None`)
        primal_attention_coeffs_thresholds (list of float/None, optional):
            Either None or a list of float values.
            - If not None, the value of the i-th element represents the
              threshold used in the pooling layer in the i-th
              `DualPrimalDownConv`/`DualPrimalResDownConv` block to pool primal
              edges for which the attention coefficient is above/below this
              value, depending on the argument
              `use_decreasing_attention_coefficients`;
            For each valid i, at most one of `num_primal_edges_to_keep[i]`
            `fractions_primal_edges_to_keep[i]` and
            `primal_attention_coeffs_thresholds[i]` can be not None. If
            `num_primal_edges_to_keep`, `fractions_primal_edges_to_keep` and
            `primal_attention_coeffs_thresholds` are all None, no pooling layers
            are inserted in the network. (default: :obj:`None`)
        num_res_blocks (int, optional): Number of residual blocks per each
            `DualPrimalDownConv`/`DualPrimalResDownConv` block in the encoder
            (cf. `num_skips` in `DualPrimalDownConv`/`DualPrimalResDownConv`).
            (default: :obj:`3`).
        heads_encoder (int, optional): Number of attention heads associated to
            each of the primal and dual convolutional layers in the
            `DualPrimalDownConv`/`DualPrimalResDownConv` blocks in the encoder.
            (default: :obj:`1`)
        heads_decoder (int, optional): Number of attention heads associated to
            each of the primal and dual convolutional layers in the decoder.
            Note: the additional `DualPrimalConv` layer (cf. arguments
            `conv_primal_out_res` and `conv_dual_out_res`) has always a single
            attention head. (default: :obj:`1`)
        concat_primal, concat_dual (bool, optional): If set to :obj:`False`, the
            attention heads associated respectively to the primal and to the
            dual convolutional layers in the `DualPrimalDownConv`/
            `DualPrimalResDownConv` blocks are averaged instead of concatenated.
            Note: in the additional `DualPrimalDownConv`/`DualPrimalResDownConv`
            block (cf. arguments `conv_primal_out_res` and `conv_dual_out_res`)
            attention heads are always averaged. (default: :obj:`True`)
        negative_slope_primal, negative_slope_dual (float, optional): LeakyReLU
            angle of the negative slope, respectively for the layers associated
            to the primal graph and for the layers associated to the dual graph.
            (default: :obj:`0.2`)
        dropout_primal, dropout_dual (float, optional): Dropout probability of
            the normalized attention coefficients which exposes each node,
            respectively of the primal and of the dual convolutional layers in
            the `DualPrimalDownConv`/`DualPrimalResDownConv` blocks, to a
            stochastically sampled neighborhood during training.
            (default: :obj:`0`)
        bias_primal, bias_dual (bool, optional): If set to :obj:`False`, the
            layers associated respectively to the primal and to the dual graph
            will not learn an additive bias. (default: :obj:`False`)
        add_self_loops_to_dual_graph (bool, optional): If set to :obj:`True`,
            regular graph-attention convolutional layers are instantiated as
            dual convolutional layers in the `DualPrimalDownConv`/
            `DualPrimalResDownConv` blocks, thus self-loops are added to the
            dual graph. Furthermore, self-loops are added to dual graphs
            generated by the pooling layers in the `DualPrimalDownConv`/
            `DualPrimalResDownConv` blocks. If set to :obj:`False` instead, a
            modified version of the graph-attention convolutional layer is
            instantiated for the dual convolutional layers in the
            `DualPrimalDownConv`/`DualPrimalResDownConv` blocks, with no
            addition of self-loops to the latter, and no self loops are added to
            the dual graphs generated by the pooling layers in the
            `DualPrimalDownConv`/`DualPrimalResDownConv` blocks.
            (default: :obj:`False`)
        allow_pooling_consecutive_edges (bool, optional): If True, no
            restrictions are put on the primal edges that can be pooled by the
            pooling layers in the model. If False, a primal edge can only be
            pooled if no primal nodes to which it belongs have been pooled
            previously. Setting this argument to False is only compatible with
            top-K pooling (cf. arguments `num_primal_edges_to_keep` and
            `fraction_primal_edges_to_keep`). (default: :obj:`True`)
        aggr_primal_features_pooling (str, optional): Parameter of the optional
            pooling layer. If 'mean'/'add', the feature of each new primal node
            after pooling is obtained by respectively averaging and summing
            the features of the primal nodes that get merged into that node.
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
            layer in the j-th `DualPrimalDownConv`/`DualPrimalResDownConv` block
            in the network. (default: :obj:`True`)
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
            layer in the j-th `DualPrimalDownConv`/`DualPrimalResDownConv` block
            in the network. (default: :obj:`False`)

    Attributes:
        bn_primal (torch_geometric.nn.norm.BatchNorm): Batch normalization layer
            that follows the primal layer of the final convolutional layer (cf.
            attribute `final_conv`).
        bn_primal_after_encoder (torch_geometric.nn.norm.BatchNorm): Batch
            normalization layer that follows the primal layer of the
            convolutional layer after the encoder (cf. attribute
            `conv_after_encoder`).
        bn_dual_after_encoder (torch_geometric.nn.norm.BatchNorm): Batch
            normalization layer that follows the dual layer of the convolutional
            layer after the encoder (cf. attribute `conv_after_encoder`).
        conv_after_encoder (pd_mesh_net.nn.conv.DualPrimalConv): Dual-primal
            convolution layer that follows the encoder and precedes the decoder
            (cf. arguments `conv_primal_out_res`/`conv_dual_out_res`).
        encoder (pd_mesh_net.models.shape_segmentation.DualPrimalMeshSegmenter):
            Encoder part of the network.        
        final_conv (pd_mesh_net.nn.conv.DualPrimalConv): Final dual-primal
            convolution layer, with a number of output primal channels equal to
            the number of classes in the dataset (cf. argument `num_classes`)
            and a number of output dual channelss equal to the number of output
            dual channels of the previous layer (cf. arguments
            `conv_primal_out_res`/`conv_dual_out_res`).
        _indices_upconv_layers (list): If the element `i` is in this list, then
            the i-th block of the encoder contains a pooling layer, and an
            up-convolution block with the same index is added to the decoder.
        ---
        upconv{i} (pd_mesh_net.models.up_conv.DualPrimalUpConv), for each `i`
            such that `self.encoder.block{i}` contains a pooling layer: `i`-th
            up-convolution block.
    """

    def __init__(self,
                 in_channels_primal,
                 in_channels_dual,
                 conv_primal_out_res,
                 conv_dual_out_res,
                 num_classes,
                 single_dual_nodes,
                 undirected_dual_edges,
                 use_dual_primal_res_down_conv_blocks=False,
                 norm_layer_type='group_norm',
                 num_groups_norm_layer=16,
                 num_primal_edges_to_keep=None,
                 fractions_primal_edges_to_keep=None,
                 primal_attention_coeffs_thresholds=None,
                 num_res_blocks=3,
                 heads_encoder=1,
                 heads_decoder=1,
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

        super(DualPrimalUNetMeshSegmenter, self).__init__()
        # Save input parameters.
        self.__input_parameters = {
            k: v for k, v in locals().items() if (k[0] != '_' and k != 'self')
        }

        self.__return_node_to_cluster = return_node_to_cluster
        self.__log_ratios_new_old_primal_nodes = log_ratios_new_old_primal_nodes
        self.__log_ratios_new_old_primal_edges = log_ratios_new_old_primal_edges

        # Add encoder.
        self.__concat_primal_encoder = concat_primal
        self.__concat_dual_encoder = concat_dual
        self.__heads_encoder = heads_encoder
        self.encoder = DualPrimalMeshSegmenter(
            in_channels_primal=in_channels_primal,
            in_channels_dual=in_channels_dual,
            conv_primal_out_res=conv_primal_out_res[:-1],
            conv_dual_out_res=conv_dual_out_res[:-1],
            num_classes=num_classes,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            use_dual_primal_res_down_conv_blocks=
            use_dual_primal_res_down_conv_blocks,
            norm_layer_type=norm_layer_type,
            num_groups_norm_layer=num_groups_norm_layer,
            num_primal_edges_to_keep=num_primal_edges_to_keep,
            fractions_primal_edges_to_keep=fractions_primal_edges_to_keep,
            primal_attention_coeffs_thresholds=
            primal_attention_coeffs_thresholds,
            num_res_blocks=num_res_blocks,
            heads=heads_encoder,
            concat_primal=concat_primal,
            concat_dual=concat_dual,
            negative_slope_primal=negative_slope_primal,
            negative_slope_dual=negative_slope_dual,
            dropout_primal=dropout_primal,
            dropout_dual=dropout_dual,
            bias_primal=bias_primal,
            bias_dual=bias_dual,
            add_self_loops_to_dual_graph=add_self_loops_to_dual_graph,
            allow_pooling_consecutive_edges=allow_pooling_consecutive_edges,
            aggr_primal_features_pooling=aggr_primal_features_pooling,
            aggr_dual_features_pooling=aggr_dual_features_pooling,
            use_decreasing_attention_coefficients=
            use_decreasing_attention_coefficients,
            return_node_to_cluster=return_node_to_cluster,
            log_ratios_new_old_primal_nodes=log_ratios_new_old_primal_nodes,
            log_ratios_new_old_primal_edges=log_ratios_new_old_primal_edges,
            return_info_for_unpooling=True,
            do_not_add_final_block=True)
        # Save internally the indices of the pooling layers, required by
        # BaseTrainingJob.
        self._indices_pooling_layers = self.encoder._indices_pooling_layers

        # Add DualPrimalConv layer after the encoder.
        out_channels_primal_considering_heads = conv_primal_out_res[-2]
        if (concat_primal):
            out_channels_primal_considering_heads *= heads_encoder
        out_channels_dual_considering_heads = conv_dual_out_res[-2]
        if (concat_dual):
            out_channels_dual_considering_heads *= heads_encoder
        self.conv_after_encoder = DualPrimalConv(
            in_channels_primal=out_channels_primal_considering_heads,
            in_channels_dual=out_channels_dual_considering_heads,
            out_channels_primal=conv_primal_out_res[-1],
            out_channels_dual=conv_dual_out_res[-1],
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            heads=heads_encoder,
            concat_primal=concat_primal,
            concat_dual=concat_dual,
            negative_slope_primal=negative_slope_primal,
            negative_slope_dual=negative_slope_dual,
            dropout_primal=dropout_primal,
            dropout_dual=dropout_dual,
            bias_primal=bias_primal,
            bias_dual=bias_dual,
            add_self_loops_to_dual_graph=add_self_loops_to_dual_graph)
        out_channels_primal_considering_heads = conv_primal_out_res[-1]
        if (concat_primal):
            out_channels_primal_considering_heads *= heads_encoder
        out_channels_dual_considering_heads = conv_dual_out_res[-1]
        if (concat_dual):
            out_channels_dual_considering_heads *= heads_encoder
        self.bn_primal_after_encoder = BatchNorm(
            in_channels=out_channels_primal_considering_heads)
        self.bn_dual_after_encoder = BatchNorm(
            in_channels=out_channels_dual_considering_heads)
        # Add one DualPrimalUpConv layer for each pooling layer in the encoder.
        assert (len(
            self.encoder._indices_pooling_layers) == len(conv_primal_out_res) -
                1), ("The model currently supports only encoders with all the "
                     "down-convolutional blocks containing a pooling layer.")
        self._indices_upconv_layers = []
        for idx_pooling_layer in self.encoder._indices_pooling_layers:
            setattr(
                self, f'upconv{idx_pooling_layer}',
                DualPrimalUpConv(
                    in_channels_primal=conv_primal_out_res[idx_pooling_layer +
                                                           1],
                    in_channels_dual=conv_dual_out_res[idx_pooling_layer + 1],
                    out_channels_primal=conv_primal_out_res[idx_pooling_layer],
                    out_channels_dual=conv_dual_out_res[idx_pooling_layer],
                    single_dual_nodes=single_dual_nodes,
                    undirected_dual_edges=undirected_dual_edges,
                    attention_heads_convolution=heads_decoder,
                    concat_data_from_before_pooling=True))
            self._indices_upconv_layers.append(idx_pooling_layer)
        # Add a final DualPrimalConv layer, followed by batch normalization, for
        # the prediction of the class labels.

        self.final_conv = DualPrimalConv(
            in_channels_primal=conv_primal_out_res[0],
            in_channels_dual=conv_dual_out_res[0],
            out_channels_primal=num_classes,
            out_channels_dual=conv_dual_out_res[0],
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            heads=1,
            concat_primal=False,
            concat_dual=False,
            negative_slope_primal=negative_slope_primal,
            negative_slope_dual=negative_slope_dual,
            dropout_primal=dropout_primal,
            dropout_dual=dropout_dual,
            bias_primal=bias_primal,
            bias_dual=bias_dual,
            add_self_loops_to_dual_graph=add_self_loops_to_dual_graph)
        self.bn_primal = BatchNorm(in_channels=num_classes)

    @property
    def input_parameters(self):
        return self.__input_parameters

    def forward(self, primal_graph_batch, dual_graph_batch,
                primal_edge_to_dual_node_idx_batch):
        r"""Forward pass. Retrieves unnormalized per-class scores for each face
        in the input mesh (i.e., for each node in the input primal graph).

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
            out (list of torch.Tensor of shape
                `[num_primal_nodes_batch, num_classes]`, where
                `num_primal_nodes_batch` is the number of primal nodes in the
                input batch): The i-th tensor in the list contains the per-class
                unnormalized scores associated to the i-th primal node in the
                input batch.
            log_info (namedtuple): Logging information. If at least one of the
                class arguments `return_node_to_cluster` and
                `log_ratios_new_old_primal_nodes` is True, contains the
                following attributes:
                - node_to_cluster (dict of torch.Tensor, optional): If the class
                    argument `return_node_to_cluster` is True, dict of
                    `num_pooling_layers` tensors, where `num_pooling_layers` is
                    the number of pooling layers in the encoder. Cf. method
                    `forward_encoder_only` of `DualPrimalMeshSegmenter` for more
                    details.
                - ratios_new_old_primal_nodes (dict of torch.Tensor, optional):
                    If the class argument `log_ratios_new_old_primal_nodes` is
                    True, dict of `num_pooling_layers` tensors, where
                    `num_pooling_layers` is the number of pooling layers in the
                    encoder. Cf. method `forward_encoder_only` of
                    `DualPrimalMeshSegmenter` for more details.
                - ratios_new_old_primal_edges (dict of torch.Tensor, optional):
                    If the class argument `log_ratios_new_old_primal_edges` is
                    True, dict of `num_pooling_layers` tensors, where
                    `num_pooling_layers` is the number of pooling layers in the
                    encoder. Cf. method `forward_encoder_only` of
                    `DualPrimalMeshSegmenter` for more details.
        """
        # Forward pass through the encoder.
        (primal_graph_batch, dual_graph_batch,
         primal_edge_to_dual_node_idx_batch, node_to_cluster_encoder,
         ratios_new_old_primal_nodes_encoder,
         ratios_new_old_primal_edges_encoder,
         primal_graph_batches_before_pooling, dual_graph_batches_before_pooling,
         pooling_info_per_block) = self.encoder.forward_encoder_only(
             primal_graph_batch=primal_graph_batch,
             dual_graph_batch=dual_graph_batch,
             primal_edge_to_dual_node_idx_batch=(
                 primal_edge_to_dual_node_idx_batch))
        # Forward pass through the dual-primal convolutional layer between
        # encoder and decoder.
        (primal_graph_batch.x, dual_graph_batch.x) = self.conv_after_encoder(
            x_primal=primal_graph_batch.x,
            x_dual=dual_graph_batch.x,
            edge_index_primal=primal_graph_batch.edge_index,
            edge_index_dual=dual_graph_batch.edge_index,
            primal_edge_to_dual_node_idx=primal_edge_to_dual_node_idx_batch)
        primal_graph_batch.x = F.relu(
            self.bn_primal_after_encoder(primal_graph_batch.x))
        dual_graph_batch.x = F.relu(
            self.bn_dual_after_encoder(dual_graph_batch.x))
        # - If multiple heads were used with concatenation in the encoder,
        #   average them before using the output features as input to the
        #   decoder.
        if (self.__heads_encoder > 1):
            if (self.__concat_primal_encoder):
                num_channels_primal_encoder = (
                    primal_graph_batch.num_node_features //
                    self.__heads_encoder)
                primal_graph_batch.x = scatter_mean(
                    primal_graph_batch.x,
                    torch.arange(num_channels_primal_encoder,
                                 device=primal_graph_batch.x.device).repeat(
                                     self.__heads_encoder))
            if (self.__concat_dual_encoder):
                num_channels_dual_encoder = (
                    dual_graph_batch.num_node_features // self.__heads_encoder)
                dual_graph_batch.x = scatter_mean(
                    dual_graph_batch.x,
                    torch.arange(num_channels_dual_encoder,
                                 device=dual_graph_batch.x.device).repeat(
                                     self.__heads_encoder))

        # Forward pass through each of the DualPrimalUpConv layers.
        for idx_upconv_layer in self.encoder._indices_pooling_layers[::-1]:
            # - If multiple heads were used with concatenation in the encoder,
            #   average them before concatenating the features from the encoder
            #   to those from the decoder.
            if (self.__heads_encoder > 1):
                if (self.__concat_primal_encoder):
                    num_channels_primal_encoder = (
                        primal_graph_batches_before_pooling[idx_upconv_layer].
                        num_node_features // self.__heads_encoder)
                    primal_graph_batches_before_pooling[
                        idx_upconv_layer].x = scatter_mean(
                            primal_graph_batches_before_pooling[
                                idx_upconv_layer].x,
                            torch.arange(
                                num_channels_primal_encoder,
                                device=primal_graph_batches_before_pooling[
                                    idx_upconv_layer].x.device).repeat(
                                        self.__heads_encoder))
                if (self.__concat_dual_encoder):
                    num_channels_dual_encoder = (
                        dual_graph_batches_before_pooling[idx_upconv_layer].
                        num_node_features // self.__heads_encoder)
                    dual_graph_batches_before_pooling[
                        idx_upconv_layer].x = scatter_mean(
                            dual_graph_batches_before_pooling[idx_upconv_layer].
                            x,
                            torch.arange(
                                num_channels_dual_encoder,
                                device=dual_graph_batches_before_pooling[
                                    idx_upconv_layer].x.device).repeat(
                                        self.__heads_encoder))
            (primal_graph_batch, dual_graph_batch,
             primal_edge_to_dual_node_idx_batch
            ) = getattr(
                self, f'upconv{idx_upconv_layer}'
            )(primal_graph_batch=primal_graph_batch,
              dual_graph_batch=dual_graph_batch,
              primal_edge_to_dual_node_idx_batch=(
                  primal_edge_to_dual_node_idx_batch),
              pooling_log=pooling_info_per_block[idx_upconv_layer],
              primal_graph_batch_before_pooling=
              primal_graph_batches_before_pooling[idx_upconv_layer],
              dual_graph_batch_before_pooling=dual_graph_batches_before_pooling[
                  idx_upconv_layer])
        # Forward pass through the final dual-primal convolutional layer.
        (primal_graph_batch.x, dual_graph_batch.x) = self.final_conv(
            x_primal=primal_graph_batch.x,
            x_dual=dual_graph_batch.x,
            edge_index_primal=primal_graph_batch.edge_index,
            edge_index_dual=dual_graph_batch.edge_index,
            primal_edge_to_dual_node_idx=primal_edge_to_dual_node_idx_batch)
        # Forward pass through batch-normalization layer.
        primal_graph_batch.x = self.bn_primal(primal_graph_batch.x)

        out = primal_graph_batch.x

        #NOTE: The output features represent unnormalized per-class scores and
        # should therefore be soft-maxed and, e.g., undergo negative log (one
        # can combine the two operations with cross-entropy loss).

        # Set up the logging information associated to the encoder part of the
        # network.
        log_info = PoolingLogMultipleLayers(
            ratios_new_old_primal_nodes=ratios_new_old_primal_nodes_encoder,
            ratios_new_old_primal_edges=ratios_new_old_primal_edges_encoder,
            node_to_cluster=node_to_cluster_encoder
            if self.__return_node_to_cluster else None)

        return out, log_info
