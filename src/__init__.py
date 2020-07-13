from catalyst.dl import SupervisedRunner as Runner
from catalyst.dl import registry

from src.symmetric_lovasz_loss import SymmetricLovaszLoss

from .callbacks import (
    CustomDiceCallback,
    CustomInferCallback,
    PostprocessingCallback,
    PseudoLabelsCallback,
    CheckpointLoader,
)
from .experiment import Experiment
from .losses import BCEDiceLossCustom

registry.Criterion(BCEDiceLossCustom)
registry.Criterion(SymmetricLovaszLoss)

registry.Callback(CustomDiceCallback)
registry.Callback(PostprocessingCallback)
registry.Callback(CustomInferCallback)
registry.Callback(PseudoLabelsCallback)
registry.Callback(CheckpointLoader)
