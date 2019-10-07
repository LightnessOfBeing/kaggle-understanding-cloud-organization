import cv2

from catalyst.dl import Callback, CallbackOrder, RunnerState
import numpy as np

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