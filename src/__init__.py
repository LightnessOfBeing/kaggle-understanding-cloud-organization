from catalyst.dl import registry
from catalyst.dl import SupervisedRunner as Runner
from .experiment import Experiment

from src.symmetric_lovasz_loss import SymmetricLovaszLoss
from catalyst.contrib.nn import BCEDiceLoss

registry.Criterion(BCEDiceLoss)
registry.Criterion(SymmetricLovaszLoss)

