from torch.nn import CrossEntropyLoss
import unittest

from pd_mesh_net.utils import create_loss


class TestCreateLoss(unittest.TestCase):

    def test_create_classification_loss(self):
        # Create the loss.
        loss_params = {'task_type': 'classification'}
        loss = create_loss(**loss_params)
        self.assertTrue(isinstance(loss, CrossEntropyLoss))

    def test_create_loss_for_nonexistent_task(self):
        # Create the loss.
        loss_params = {'task_type': 'nonexistent_task'}
        self.assertRaises(KeyError, create_loss, **loss_params)
