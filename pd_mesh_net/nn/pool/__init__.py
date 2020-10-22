from collections import namedtuple

# Similarly to what done in `torch_geometric.nn.pool.EdgePooling`, here
# information is stored that, for a certain pooling layer, maps the input graphs
# to the graphs formed after pooling. Optionally, stores the ratio between the
# number of new and old primal nodes/edges after pooling
PoolingInfo = namedtuple('PoolingInfo', [
    'old_primal_node_to_new_one', 'old_dual_node_to_new_one',
    'old_primal_edge_index', 'old_dual_edge_index', 'old_primal_graph_batch',
    'old_dual_graph_batch', 'old_primal_edge_to_dual_node_index',
    'ratio_new_old_primal_nodes', 'ratio_new_old_primal_edges'
])

# Data structure used to gather information about multiple pooling layers in a
# certain architecture. In particular, the node-to-cluster list is returned, and
# for logging purposes, the ratios between the number of new and old primal
# nodes/edges after pooling
PoolingLogMultipleLayers = namedtuple('PoolingLogMultipleLayers', [
    'node_to_cluster', 'ratios_new_old_primal_nodes',
    'ratios_new_old_primal_edges'
])

from .dual_primal_edge_pool import DualPrimalEdgePooling
__all__ = ['DualPrimalEdgePooling', 'PoolingInfo', 'PoolingLogMultipleLayers']