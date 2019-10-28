from typing import Dict

import cv2

from catalyst.dl import Callback, CallbackOrder, RunnerState, CheckpointCallback, MetricCallback
import numpy as np
from catalyst.utils import save_checkpoint


class CustomSegmentationInferCallback(Callback):
    def __init__(self, return_valid: bool = False):
        super().__init__(CallbackOrder.Internal)
        self.valid_masks = []
        self.probabilities = np.zeros((2220, 350, 525))
        self.return_valid = return_valid

    def on_batch_end(self, state: RunnerState):
        image, mask = state.input
        output = state.output["logits"]
        if self.return_valid:
            for m in mask:
                if m.shape != (350, 525):
                    m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                self.valid_masks.append(m)

        for j, probability in enumerate(output):
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            self.probabilities[j, :, :] = probability


class CustomCheckpointCallback(CheckpointCallback):
    def process_checkpoint(
            self,
            logdir: str,
            checkpoint: Dict,
            is_best: bool,
            main_metric: str = "loss",
            minimize_metric: bool = True
    ):
        exclude = ["criterion", "optimizer", "scheduler"]
        checkpoint = {
            key: value
            for key, value in checkpoint.items()
            if all(z not in key for z in exclude)
        }
        suffix = self.get_checkpoint_suffix(checkpoint)
        suffix = f"{suffix}.exception_"

        filepath = save_checkpoint(
            checkpoint=checkpoint,
            logdir=f"{logdir}/checkpoints/",
            suffix=suffix,
            is_best=is_best,
            is_last=True
        )

        valid_metrics = checkpoint["valid_metrics"]
        checkpoint_metric = valid_metrics[main_metric]
        self.top_best_metrics.append(
            (filepath, checkpoint_metric, valid_metrics)
        )
        self.truncate_checkpoints(minimize_metric=minimize_metric)

        metrics = self.get_metric(valid_metrics)
        self.save_metric(logdir, metrics)


def my_dice(img1, img2, **kwargs):
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()

    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)
    if img1.sum() + img2.sum() == 0: return 1
    intersection = np.logical_and(img1, img2)
    return 2. * intersection.sum() / (img1.sum() + img2.sum())

class CustomDiceCallback(MetricCallback):
    """
    Dice metric callback.
    """
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "dice",
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid"
    ):
        """
        :param input_key: input key to use for dice calculation;
            specifies our `y_true`.
        :param output_key: output key to use for dice calculation;
            specifies our `y_pred`.
        """
        super().__init__(
            prefix="custom_dice_kirill",
            metric_fn=my_dice,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation
        )