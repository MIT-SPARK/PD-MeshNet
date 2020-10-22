import torch
import warnings


def create_optimizer(optimizer_type, network_parameters, **optimizer_params):
    r"""Creates an instance of the input optimizer with the input parameters.

    Args:
        optimizer_type (str): Name that identifies the optimizer. Valid values
            are: `adam`.
        network_parameters (generator of torch.nn.parameter.Parameter): Network
            parameters that the optimizer should optimize on.
        ...
        Optional parameters of the optimizer.

    Returns:
        optimizer (torch.optim.optimizer.Optimizer): The instance of the
            optimizer with the input parameters.
    """
    if (optimizer_type == 'adam'):
        # Only keep the valid optimizer parameters.
        valid_optimizer_params = {}
        for param, param_value in optimizer_params.items():
            if (param in ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad']):
                valid_optimizer_params[param] = param_value
            else:
                warnings.warn(
                    f"Ignoring parameter '{param}', invalid for optimizer Adam."
                )
        optimizer = torch.optim.Adam(params=network_parameters,
                                     **valid_optimizer_params)
    else:
        raise KeyError("No known optimizer can be created with the name "
                       f"'{optimizer_type}'.")

    return optimizer