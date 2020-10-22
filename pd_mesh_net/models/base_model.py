from abc import ABC, abstractmethod
import torch


class BaseDualPrimalModel(ABC, torch.nn.Module):
    r"""Base model class, used to define pure virtual methods that each model
    should implement.

    Args:
        None.
    """

    def __init__(self):
        super(BaseDualPrimalModel, self).__init__()

    @property
    @abstractmethod
    def input_parameters(self):
        pass