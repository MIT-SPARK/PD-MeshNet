import inspect
import torch
import warnings

from pd_mesh_net.models import (DualPrimalMeshClassifier,
                                DualPrimalMeshSegmenter,
                                DualPrimalUNetMeshSegmenter)


def create_model(model_name, should_initialize_weights, **model_params):
    r"""Creates an instance of the input model with the input parameters.

    Args:
        model_name (str): Name that identifies the model. Valid values are:
            `mesh_classifier`, 'mesh_segmenter', 'unet_mesh_segmenter'.
        should_initialize_weights (bool): Whether or not to perform weights
            initialization. If True, parameters `weight_initialization_type` and
            `weight_initialization_gain` are also required.  
        ...
        Optional parameters of the models.

    Returns:
        model (torch.nn.Module): The instance of the model with the input
            parameters.
    """
    if (model_name == 'mesh_classifier'):
        model_class = DualPrimalMeshClassifier
    elif (model_name == 'mesh_segmenter'):
        model_class = DualPrimalMeshSegmenter
    elif (model_name == 'unet_mesh_segmenter'):
        model_class = DualPrimalUNetMeshSegmenter
    else:
        raise KeyError(
            f"No known model can be created with the name '{model_name}'.")

    # Only keep the valid model parameters.
    valid_model_params = {}
    possible_valid_params = [
        p for p in inspect.getfullargspec(model_class).args
        if p not in ['self']
    ]
    for param, param_value in model_params.items():
        if (param in possible_valid_params):
            valid_model_params[param] = param_value
        else:
            if (param not in [
                    'weight_initialization_type', 'weight_initialization_gain'
            ]):
                warnings.warn(
                    f"Ignoring parameter '{param}', invalid for model "
                    f"'{model_class.__name__}'.")

    # Create model.
    model = model_class(**valid_model_params)

    # Optionally initialize model weights.
    if (should_initialize_weights and
            'weight_initialization_type' in model_params and
            'weight_initialization_gain' in model_params):
        initialize_model_weights(
            model=model,
            initialization_type=model_params['weight_initialization_type'],
            initialization_gain=model_params['weight_initialization_gain'])

    return model


def initialize_model_weights(model, initialization_type, initialization_gain):
    """ Initializes the weights of the input network.
    Modified from MeshCNN (https://github.com/ranahanocka/MeshCNN/).

    Args:
        model (torch.nn.Module): Model used.
        initialization_type (str): One of the following:
            - 'kaiming': 'He initialization' is used, cf.,
                https://pytorch.org/docs/stable/nn.init.html.
            - 'normal': Weights are drawn from a normal distribution with mean
                0 and variance `init_gain`.
            - 'orthogonal': Weights are initialization with a (semi) orthogonal
                matrix, with scaling factor `init_gain`, cf.
                https://pytorch.org/docs/stable/nn.init.html.
            - 'xavier': 'Glorot initialization' is used, with gain `init_gain`,
                cf. https://pytorch.org/docs/stable/nn.init.html.
        initialization_gain (float): Factor for weight initialization, cf.
            argument `initialization_type`.
    
    Returns:
        None.
    """

    def initialize_module(module):
        """ Initializes the weights of the linear and batch normalization
        layers. Convolutional layers are automatically initialized (as they are
        derived classed of `torch_geometric.nn.conv.GATConv`).
        Modified from MeshCNN (https://github.com/ranahanocka/MeshCNN/).

        Args:
            module (torch.nn.Module): Submodule of the network to which the
                initialization function should be applied.

        Returns:
            None.
        """
        class_name = module.__class__.__name__
        if (hasattr(module, 'weight') and class_name.find('Linear') != -1):
            if (initialization_type == 'normal'):
                torch.nn.init.normal_(module.weight.data, 0.0,
                                      initialization_gain)
            elif (initialization_type == 'xavier'):
                torch.nn.init.xavier_normal_(module.weight.data,
                                             gain=initialization_gain)
            elif (initialization_type == 'kaiming'):
                torch.nn.init.kaiming_normal_(module.weight.data,
                                              a=0,
                                              mode='fan_in')
            elif (initialization_type == 'orthogonal'):
                torch.nn.init.orthogonal_(module.weight.data,
                                          gain=initialization_gain)
            else:
                raise NotImplementedError(
                    f"Initialization method {initialization_type} is not "
                    "implemented.")
        elif (class_name.find('BatchNorm') != -1):
            torch.nn.init.normal_(module.weight.data, 1.0, initialization_gain)
            torch.nn.init.constant_(module.bias.data, 0.0)

    # Recursively apply the initialization function to all the submodules in the
    # network.
    model.apply(initialize_module)