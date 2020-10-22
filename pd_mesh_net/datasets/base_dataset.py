from abc import ABC, abstractmethod
from torch_geometric.data import Dataset


class BaseDualPrimalDataset(ABC, Dataset):
    r"""Base dataset class, used to define pure virtual methods that each
    dataset should implement.

    Args:
        None.
    """

    def __init__(self, root, transform, pre_transform, pre_filter):
        super(BaseDualPrimalDataset, self).__init__(root, transform,
                                                    pre_transform, pre_filter)

    @property
    @abstractmethod
    def input_parameters(self):
        pass