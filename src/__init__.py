from catalyst.dl import registry
from catalyst.dl import SupervisedRunner as Runner

from .callbacks import CustomDiceCallback, CustomSegmentationInferCallback
from .experiment import Experiment

from src.symmetric_lovasz_loss import SymmetricLovaszLoss
from catalyst.contrib.nn import BCEDiceLoss

registry.Criterion(BCEDiceLoss)
registry.Criterion(SymmetricLovaszLoss)

registry.Callback(CustomDiceCallback)
registry.Callback(CustomSegmentationInferCallback)

