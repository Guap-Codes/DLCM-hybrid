# python/dlcm/__init__.py
"""
Dynamic Large Concept Models (DLCM) - Python Control Plane
"""

from .models.dlcm import DLCMModel, DLCMConfig
from .training.global_parser import GlobalParser
from .training.decoupled_mup import get_decoupled_mup_optimizer, initialize_mup_weights

# NOTE: DLCMTrainingModule is intentionally NOT imported here. 
# This prevents pytorch_lightning from loading when you only want to test model shapes.
# When you need the trainer, import it directly: 
# from dlcm.training.trainer import DLCMTrainingModule

__version__ = "0.1.0"
__all__ = [
    "DLCMModel", 
    "DLCMConfig", 
    "GlobalParser",
    "get_decoupled_mup_optimizer",
    "initialize_mup_weights"
]


"""
Dynamic Large Concept Models (DLCM) - Python Control Plane

from .models.dlcm import DLCMModel, DLCMConfig
from .training.trainer import DLCMTrainingModule
from .training.global_parser import GlobalParser
from .training.decoupled_mup import get_decoupled_mup_optimizer, initialize_mup_weights

__version__ = "0.1.0"
__all__ = [
    "DLCMModel", 
    "DLCMConfig", 
    "DLCMTrainingModule",
    "GlobalParser",
    "get_decoupled_mup_optimizer",
    "initialize_mup_weights"
]

"""
