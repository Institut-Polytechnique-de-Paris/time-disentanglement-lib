from typing import Tuple, Union

from pydantic.dataclasses import dataclass

from XGen.config import BaseConfig


@dataclass
class BaseNFConfig(BaseConfig):
    """This is the Base Normalizing Flow config instance.

    Parameters:
        input_dim (tuple): The input data dimension. Default: None.
    """

    input_dim: Union[Tuple[int, ...], None] = None
