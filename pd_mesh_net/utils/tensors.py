import torch


class TensorClusters():
    r"""Implements a data structure that holds a set of 1-D tensors. If an input
    tensor vector shares an element with tensor already in the set, it gets
    merged into it; otherwise, the tensor is added to the set as a new element.

    Args:
        store_tensor_indices_per_cluster (bool, optional): If True, stores the
            indices of the tensors that are added to the data structure, where
            the indices are non-negative integer values that are incremented
            each time a new add operation is called. (default: :obj:`False`)

    Attributes:
        None.
    """

    def __init__(self, store_tensor_indices_per_cluster=False):
        self.__clusters = []
        if (store_tensor_indices_per_cluster):
            self.__tensor_indices_per_cluster = []
            self.__last_tensor_index = 0
        else:
            self.__tensor_indices_per_cluster = None

    @property
    def clusters(self):
        r"""Returns the 'clusters', i.e., the merged tensors, in the data
        structure. If the class input argument
        `store_tensor_indices_per_cluster` was set to True, also the indices of
        the tensors that were added to the data structure are returned (cf.
        argument `store_tensor_indices_per_cluster`).
        """
        if (self.__tensor_indices_per_cluster is not None):
            return self.__clusters, self.__tensor_indices_per_cluster
        else:
            return self.__clusters

    def add(self, element):
        r"""Adds an input 1-D tensor to the data structure.

        Args:
            element (torch.Tensor of shape :obj:`[vector_length,]`, where
                `vector_length` is the size of the vector): Element to add to
                the union-find data structure.

        Returns:
            None.
        """
        assert (torch.is_tensor(element))
        assert (element.dim() == 1)

        # Iterate over the existing tensor sets.
        merged_new_element_in_other_sets = False
        # These variable contain the indices of the previously-existing tensors
        # that should or should not be merged to the input tensor.
        previous_clusters_to_merge = []
        previous_clusters_not_to_merge = []
        for cluster_idx, cluster in enumerate(self.__clusters):
            # Concatenate the input tensor to the one already in the set and
            # remove duplicates.
            union_tensor = torch.cat([cluster, element]).unique()

            if (len(union_tensor) < len(cluster) + len(element)):
                # If at least one element is in common between the
                # previously-existing tensor and the input one, set them to be
                # merged.
                previous_clusters_to_merge.append(cluster_idx)
                merged_new_element_in_other_sets = True
            else:
                # If no elements are in common, set that the previously-existing
                # tensor should not be used for merging.
                previous_clusters_not_to_merge.append(cluster_idx)

        if (merged_new_element_in_other_sets):
            # Merge the previously-existing tensors that should be merged with
            # the input tensor.
            new_cluster = torch.cat([
                element, *[
                    self.__clusters[cluster_idx]
                    for cluster_idx in previous_clusters_to_merge
                ]
            ]).unique()
            self.__clusters = [
                new_cluster, *[
                    self.__clusters[cluster_idx]
                    for cluster_idx in previous_clusters_not_to_merge
                ]
            ]
            if (self.__tensor_indices_per_cluster is not None):
                new_tensor_indices_per_cluster = torch.cat([
                    torch.LongTensor([self.__last_tensor_index]), *[
                        self.__tensor_indices_per_cluster[cluster_idx]
                        for cluster_idx in previous_clusters_to_merge
                    ]
                ]).sort()[0]
                self.__tensor_indices_per_cluster = [
                    new_tensor_indices_per_cluster, *[
                        self.__tensor_indices_per_cluster[cluster_idx]
                        for cluster_idx in previous_clusters_not_to_merge
                    ]
                ]
        else:
            # If the new element could not be merged to any previously-existing
            # element in the set, add it to the set as a new element.
            self.__clusters.append(element)
            if (self.__tensor_indices_per_cluster is not None):
                self.__tensor_indices_per_cluster.append(
                    torch.LongTensor([self.__last_tensor_index]))

        if (self.__tensor_indices_per_cluster is not None):
            self.__last_tensor_index += 1


class NodeClustersWithUnionFind():
    r"""Implements a data structure that holds clusters of nodes. Edges are
    given as inputs, and their node endpoints are assigned to the same cluster.
    If the nodes were already part of a cluster, the clusters to which they
    belonged are merged. An underlying union-find data structure on the nodes is
    used.

    Args:
        will_input_tensors (bool): If True, it will be assumed that the input
            edges are tensors; otherwise, it will be assumed that they are
            lists.
        num_nodes (int, optional): If not None, number of nodes in the graph
            from which the edges are taken. It will be assumed that all the
            nodes that are endpoints of the edges being inputted have an index
            between 0 and `num_nodes` - 1. Only required if argument
            `will_input_tensors` is True. (default: :obj:`None`)
        device (torch.device, optional): If not None, type of device on which
            the underlying tensors should be allocated, in case the input edges
            are tensors. Only required if argument `will_input_tensors` is True.
            (default: :obj:`None`)

    Attributes:
        None.
    """

    def __init__(self, will_input_tensors, num_nodes=None, device=None):
        if (will_input_tensors):
            assert (
                isinstance(num_nodes, int) and isinstance(device, torch.device)
            ), ("Please specify the total number of nodes in the graph and the "
                "device on which the tensors should be allocated.")
            assert (num_nodes > 0)
        self.__will_input_tensors = will_input_tensors
        # Initially, each node is assigned to its own cluster.
        if (will_input_tensors):
            self.__device = device
            self.__parent = torch.arange(num_nodes,
                                         dtype=torch.long,
                                         device=device)
            self.__num_nodes = num_nodes
            # Store which nodes have been actually inserted in a cluster with
            # another node.
            self.__was_node_found = torch.zeros(num_nodes,
                                                dtype=torch.bool,
                                                device=device)
            self.add_nodes_from_edge = self.__add_nodes_from_edge_tensor
        else:
            self.__parent = {}
            self.add_nodes_from_edge = self.__add_nodes_from_edge_list
        # We implement the union operation by rank.
        if (will_input_tensors):
            self.__rank = torch.ones(num_nodes, dtype=torch.long, device=device)
        else:
            self.__rank = {}

    @property
    def clusters(self):
        r"""Returns the clusters of nodes.
        """
        if (self.__will_input_tensors):
            return self.__clusters_tensor()
        else:
            return self.__clusters_list()

    def __clusters_tensor(self):
        # Select only the nodes that were actually found.
        indices_nodes_to_consider = torch.arange(
            self.__num_nodes, dtype=torch.long,
            device=self.__device)[self.__was_node_found].view(-1, 1)
        # Form clusters.
        node_clusters = []
        node_to_cluster_idx = -torch.ones(
            self.__num_nodes, dtype=torch.long, device=self.__device)
        for node_idx in indices_nodes_to_consider:
            parent_node_idx = self.__find(node_idx)
            cluster_idx_parent_node = node_to_cluster_idx[parent_node_idx]
            if (cluster_idx_parent_node == -1):
                # If the parent node of the current node has not yet been
                # assigned to any cluster, form a new cluster, and assign the
                # root node to it too.
                node_to_cluster_idx[parent_node_idx] = len(node_clusters)
                node_clusters.append(node_idx)
            else:
                # The parent node has already been assigned to a cluster: add
                # the current node to that cluster.
                node_clusters[cluster_idx_parent_node] = torch.cat(
                    [node_clusters[cluster_idx_parent_node], node_idx], dim=0)

        return node_clusters

    def __clusters_list(self):
        node_clusters = []
        node_to_cluster_idx = dict()
        for node_in_cluster in self.__parent.keys():
            parent_node_idx = self.__find_single_node(node_in_cluster)
            if (not parent_node_idx in node_to_cluster_idx):
                # If the parent node of the current node has not yet been
                # assigned to any cluster, form a new cluster, and assign the
                # root node to it too.
                node_to_cluster_idx[parent_node_idx] = len(node_clusters)
                node_clusters.append({node_in_cluster})
            else:
                # The parent node has already been assigned to a cluster: add
                # the current node to that cluster.
                cluster_idx_parent_node = node_to_cluster_idx[parent_node_idx]
                node_clusters[cluster_idx_parent_node].add(node_in_cluster)

        return node_clusters

    def __add_nodes_from_edge_tensor(self, edge):
        r"""Adds to the data structure the nodes that are endpoints of the input
        edge.

        Args:
            edge (torch.Tensor of shape :obj:`[2,]`): Edge the endpoints of
                which should be added to the node clusters.

        Returns:
            None.
        """
        # Set the endpoint nodes as found.
        self.__was_node_found[edge] = True
        # Merge the clusters to which the nodes belong.
        self.__union_tensor(edge)

    def __add_nodes_from_edge_list(self, edge):
        r"""Adds to the data structure the nodes that are endpoints of the input
        edge.

        Args:
            edge (list of int): Edge the endpoints of which should be added to
                the node clusters.

        Returns:
            None.
        """
        # Merge the clusters to which the nodes belong.
        self.__union_list(edge)

    def __find_tensor(self, nodes):
        r"""Finds the parent cluster of one or more input nodes.

        Args:
            nodes (torch.Tensor of shape :obj:`[num_nodes,]`, where `num_nodes`
                is the number of input nodes): Nodes of which to find the parent
                cluster in the union-find set.

        Returns:
            parents (torch.Tensor of shape :obj:`[num_nodes,]`, where
                `num_nodes` is the number of input nodes): Indices of the parent
                clusters associated to the input nodes.
        """
        if (not (torch.equal(self.__parent[nodes], nodes))):
            self.__parent[nodes] = self.__find_tensor(self.__parent[nodes])

        parents = self.__parent[nodes]

        return parents

    def __find_list(self, nodes):
        r"""Finds the parent cluster of one or more input nodes.

        Args:
            nodes (list of length `num_nodes`, where `num_nodes` is the number
                of input nodes): Nodes of which to find the parent cluster in
                the union-find set.

        Returns:
            parents (list of length `num_nodes`, where `num_nodes` is the number
                of input nodes): Indices of the parent clusters associated to
                the input nodes.
        """
        parents = len(nodes) * [None]
        for idx, node in enumerate(nodes):
            if (not node in self.__parent):
                # If the node is not in any cluster yet, add it to a cluster
                # of its own.
                self.__parent[node] = node
                self.__rank[node] = 1
                parents[idx] = node
            else:
                parents[idx] = self.__find_single_node(node)

        return parents

    def __find_single_node(self, node):
        r"""Same as above, but for a single input node.

        Args:
            node (int): Node of which to find the parent cluster in the
                union-find set.
        
        Returns:
            parent (int): Index of the parent cluster associated to the input
                node.
        """
        if (not self.__parent[node] == node):
            self.__parent[node] = self.__find_single_node(self.__parent[node])

        return self.__parent[node]

    def __union_tensor(self, nodes):
        r"""Performs the union of the clusters to which two input nodes belong.

        Args:
            nodes (torch.Tensor of shape :obj:`[2,]`): Nodes the clusters of
                which should be merged.
            
        Returns:
            None.
        """
        assert (nodes.shape == (2,))
        root_node_1, root_node_2 = self.__find_tensor(nodes)

        # The two nodes are already in the same cluster, no union is necessary.
        if (torch.equal(root_node_1, root_node_2)):
            return

        # Merge the node the cluster of which has lower rank into the cluster of
        # the other node.
        if (self.__rank[root_node_1] <= self.__rank[root_node_2]):
            self.__parent[root_node_1] = root_node_2
            if (torch.equal(self.__rank[root_node_1],
                            self.__rank[root_node_2])):
                self.__rank[root_node_2] += 1
        else:
            self.__parent[root_node_2] = root_node_1
            if (torch.equal(self.__rank[root_node_1],
                            self.__rank[root_node_2])):
                self.__rank[root_node_1] += 1

    def __union_list(self, nodes):
        r"""Performs the union of the clusters to which two input nodes belong.

        Args:
            nodes (list of length 2): Nodes the clusters of which should be
                merged.
            
        Returns:
            None.
        """
        assert (len(nodes) == 2)
        root_node_1, root_node_2 = self.__find_list(nodes)

        # The two nodes are already in the same cluster, no union is necessary.
        if (root_node_1 == root_node_2):
            return

        # Merge the node the cluster of which has lower rank into the cluster of
        # the other node.
        if (self.__rank[root_node_1] <= self.__rank[root_node_2]):
            self.__parent[root_node_1] = root_node_2
            if (self.__rank[root_node_1] == self.__rank[root_node_2]):
                self.__rank[root_node_2] += 1
        else:
            self.__parent[root_node_2] = root_node_1
            if (self.__rank[root_node_1] == self.__rank[root_node_2]):
                self.__rank[root_node_1] += 1
