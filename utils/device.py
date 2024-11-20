import torch
from typing import Any, Tuple, Union


def to_device(*args: Union[torch.Tensor, torch.nn.Module]) -> Tuple[Union[torch.Tensor, torch.nn.Module], ...]:
    """
    Move the given tensors or modules to the available device (GPU if available, else CPU).

    Args:
        *args: Variable length argument list of tensors or modules to move to the device.

    Returns:
        Tuple[Union[torch.Tensor, torch.nn.Module], ...]: A tuple containing the input arguments moved to the appropriate device.

    Example:
        >>> tensor = torch.randn(2, 3)
        >>> model = MyModel()
        >>> tensor, model = to_device(tensor, model)
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return tuple(argument.to(device) for argument in args)