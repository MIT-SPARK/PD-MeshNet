import inspect
from torch.optim import lr_scheduler
import warnings


def create_lr_scheduler(scheduler_type, optimizer, **scheduler_params):
    r"""Creates an instance of the input learning-rate scheduler with the input
    parameters.
    Modified from MeshCNN (https://github.com/ranahanocka/MeshCNN/).

    Args:
        scheduler_type (str): Name that identifies the learning-rate scheduler.
            Valid values are: `lambda` (linearly-decaying learning-rate
            scheduler, using `torch.optim.LambdaLR`), `step` (`StepLR` from
            `torch.optim.lr_scheduler`), `plateau` (`ReduceLROnPlateau` from
            `torch.optim.lr_scheduler`).
        optimizer (torch.optim.optimizer.Optimizer): Optimizer wrapped by the
            learning-rate scheduler.
        ...
        Optional parameters of the learning-rate scheduler.

    Returns:
        scheduler (torch.optim.lr_scheduler._LRScheduler): The instance of the
            learning-rate scheduler with the input parameters.
    """
    if (scheduler_type == 'lambda'):
        # Slightly modified version of linearly-decaying: from epoch 1 (the
        # first one) to epoch 'last_epoch_constant_lr', the learning rate is
        # kept at its initial value (dependent on the optimizer); from epoch
        # 'last_epoch_constant_lr' to epoch 'last_epoch', the learning rate
        # decays linearly, and is becomes 0 from epoch 'last_epoch' + 1 on.
        try:
            last_epoch_constant_lr = scheduler_params['last_epoch_constant_lr']
            assert (last_epoch_constant_lr > 0)
        except KeyError:
            raise KeyError(
                "Learning-rate scheduler 'lambda' requires parameter "
                "'last_epoch_constant_lr'.")
        try:
            last_epoch = scheduler_params['last_epoch']
            assert (last_epoch >= last_epoch_constant_lr)
        except KeyError:
            raise KeyError(
                "Learning-rate scheduler 'lambda' requires parameter "
                "'last_epoch'.")

        def lr_lambda(epoch):
            # This learning rate at each epoch is given by this coefficient
            # multiplied by the original learning rate.
            # - The learning-rate schedulers internally set their initial epoch
            #   to be 0, but we start from epoch 1.
            epoch = epoch + 1
            assert (epoch >= 1)
            if (epoch <= last_epoch_constant_lr):
                return 1.0
            else:
                return max(0, (epoch - last_epoch - 1) /
                           (last_epoch_constant_lr - last_epoch - 1))

        scheduler = lr_scheduler.LambdaLR(optimizer=optimizer,
                                          lr_lambda=lr_lambda)
    else:
        if (scheduler_type == 'step'):
            scheduler_class = lr_scheduler.StepLR
        elif (scheduler_type == 'plateau'):
            scheduler_class = lr_scheduler.ReduceLROnPlateau
        else:
            raise KeyError("No known scheduler can be created with the name "
                           f"'{scheduler_type}'.")
        # Only keep the valid learning-rate scheduler parameters.
        valid_scheduler_params = {}
        possible_valid_params = [
            p for p in inspect.getfullargspec(scheduler_class).args
            if p not in ['self', 'optimizer']
        ]
        for param, param_value in scheduler_params.items():
            if (param in possible_valid_params):
                valid_scheduler_params[param] = param_value
            else:
                warnings.warn(
                    f"Ignoring parameter '{param}', invalid for learning-rate "
                    f"scheduler '{scheduler_class.__name__}'.")
        scheduler = scheduler_class(optimizer=optimizer,
                                    **valid_scheduler_params)

    return scheduler