import cv2
import numpy as np
import pandas as pd
from catalyst.core import State
from catalyst.dl import Callback, CallbackOrder, MetricCallback

from src.utils import mean_dice_coef, post_process, sigmoid, dice


class CustomSegmentationInferCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.External)
        self.valid_masks = []
        self.probabilities = np.zeros((2220, 350, 525))

    def on_stage_start(self, state: "State"):
        print("Stage 3 started!")

    def on_batch_end(self, state: "State"):
        image, mask = state.batch_in
        output = state.batch_out["logits"]
        for m in mask:
            if m.shape != (350, 525):
                m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            self.valid_masks.append(m)

        for j, probability in enumerate(output):
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            self.probabilities[j, :, :] = probability

    def on_stage_end(self, state: "State"):
        class_params = {}
        for class_id in range(4):
            print(class_id)
            attempts = []
            for t in range(0, 100, 10):
                t /= 100
                for ms in [0, 1000, 5000, 10000, 11000, 14000, 15000, 16000, 18000, 19000, 20000, 21000, 23000, 25000,
                           27000, 30000, 50000]:
                    masks = []
                    for i in range(class_id, len(self.probabilities), 4):
                        probability = self.probabilities[i]
                        predict, num_predict = post_process(sigmoid(probability), t, ms)
                        masks.append(predict)

                    d = []
                    for i, j in zip(masks, self.valid_masks[class_id::4]):
                        if (i.sum() == 0) & (j.sum() == 0):
                            d.append(1)
                        else:
                            d.append(dice(i, j))

                    attempts.append((t, ms, np.mean(d)))

            attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

            attempts_df = attempts_df.sort_values('dice', ascending=False)
            print(attempts_df.head())
            best_threshold = attempts_df['threshold'].values[0]
            best_size = attempts_df['size'].values[0]

            class_params[class_id] = (best_threshold, best_size)
            np.save('./logs/class_params.npy', class_params)

'''
class CustomDiceCallback(Callback):
    def __init__(self, input_key: str = "targets", output_key: str = "logits", prefix: str = "dice_kirill"):
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        super().__init__(CallbackOrder.Metric)

    def on_batch_end(self, state: State) -> None:
        state.batch_metrics[self.prefix] = mean_dice_coef(state.batch_out[self.output_key], state.batch_in[self.input_key])

'''
class CustomDiceCallback(MetricCallback):
    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            eps: float = 1e-7,
            threshold: float = None,
            activation: str = "Sigmoid"
    ):
        super().__init__(
            prefix="dice_kirill",
            metric_fn=mean_dice_coef,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation
        )
