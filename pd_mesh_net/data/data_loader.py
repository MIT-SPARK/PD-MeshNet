import torch

from pd_mesh_net.utils import create_dual_primal_batch


class DualPrimalDataLoader(torch.utils.data.DataLoader):
    r"""Data loader. Similarly to :class:`torch_geometric.data.DataLoader`,
    merges data objects from a dual-primal dataset to form mini-batches.

    Args:
        dataset (torch_geometric.data.Dataset): The dataset from which the data
            is loaded.
        batch_size (int, optional): Number of samples in each mini-batch.
            (default: :obj:`1`)
        shuffle (bool, optional): If True, the data is reshuffled at every
            epoch. (default: :obj:`False`)
        primal_mean, primal_std, dual_mean, dual_std (numpy array, optional):
            Mean and standard deviation of the primal-graph- and dual-graph-
            node features in the dataset, respectively. If not None, the node
            features of each graph in the batch are 'standardized' (i.e., the
            mean over all nodes in the dataset becomes 0., and the standard
            deviation becomes 1.).
            (default: :obj:`None`)
        return_sample_indices (bool, optional): If True, for each batch also the
            indices that the samples in the batch have in the dataset will be
            returned. (default: :obj:`False`)
    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 primal_mean=None,
                 primal_std=None,
                 dual_mean=None,
                 dual_std=None,
                 return_sample_indices=False,
                 **kwargs):

        def collate(batch):
            # Create the batch.
            primal_graph_list = []
            dual_graph_list = []
            primal_edge_to_dual_node_idx_list = []
            sample_indices = []
            for (primal_graph, dual_graph, primal_edge_to_dual_node_idx,
                 sample_index) in batch:
                primal_graph_list.append(primal_graph)
                dual_graph_list.append(dual_graph)
                primal_edge_to_dual_node_idx_list.append(
                    primal_edge_to_dual_node_idx)
                if (self.__return_sample_indices):
                    sample_indices.append(sample_index)
            (primal_graph_batch, dual_graph_batch,
             primal_edge_to_dual_node_idx_batch) = create_dual_primal_batch(
                 primal_graph_list,
                 dual_graph_list,
                 primal_edge_to_dual_node_idx_list,
                 primal_mean=self.__primal_mean,
                 primal_std=self.__primal_std,
                 dual_mean=self.__dual_mean,
                 dual_std=self.__dual_std)

            if (self.__return_sample_indices):
                return (primal_graph_batch, dual_graph_batch,
                        primal_edge_to_dual_node_idx_batch, sample_indices)
            else:
                return (primal_graph_batch, dual_graph_batch,
                        primal_edge_to_dual_node_idx_batch)

        self.__primal_mean = primal_mean
        self.__primal_std = primal_std
        self.__dual_mean = dual_mean
        self.__dual_std = dual_std
        self.__return_sample_indices = return_sample_indices

        if (primal_mean is None or primal_std is None or dual_mean is None or
                dual_std is None):
            print(
                "\033[93mWarning: no data standardization will be performed! "
                "Please make sure to pass all the following arguments to the "
                "data loader if you instead wish to perform standardization: "
                "'primal_mean', 'primal_std', 'dual_mean', 'dual_std'.\033[00m")

        super(DualPrimalDataLoader,
              self).__init__(dataset,
                             batch_size,
                             shuffle,
                             collate_fn=lambda batch: collate(batch),
                             **kwargs)

        # Store input parameters.
        self.__input_parameters = {}
        for k, v in locals().items():
            # Do not store the dataset as input parameter, as its parameters
            # can be stored separately.
            if (k[0] != '_' and k not in ['self', 'collate', 'dataset']):
                if (k == 'kwargs'):
                    self.__input_parameters.update(**kwargs)
                else:
                    self.__input_parameters[k] = v

    @property
    def input_parameters(self):
        return self.__input_parameters
