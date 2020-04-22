from catalyst.dl import registry
from catalyst.dl import SupervisedRunner as Runner

from .callbacks import CustomDiceCallback, PostprocessingCallback
from .experiment import Experiment

from src.symmetric_lovasz_loss import SymmetricLovaszLoss
from .losses import BCEDiceLossCustom

registry.Criterion(BCEDiceLossCustom)
registry.Criterion(SymmetricLovaszLoss)

registry.Callback(CustomDiceCallback)
registry.Callback(PostprocessingCallback)

