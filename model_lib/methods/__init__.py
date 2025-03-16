# Copyright (c) CAIRI AI Lab. All rights reserved

from .adnet import ADNet

method_maps = {
'adnet':ADNet,
}

__all__ = [
    'adnet',
]