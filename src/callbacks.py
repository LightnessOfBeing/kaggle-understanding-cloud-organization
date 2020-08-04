import os

import cv2
import numpy as np
import pandas as pd
from catalyst import utils
from catalyst.dl import InferCallback, MetricCallback, State, Callback, CallbackOrder
from pytorch_toolbelt.inference import tta

from src.utils import mask2rle, mean_dice_coef, post_process, sigmoid, single_dice_coef


class PostprocessingCallback(InferCallback):
    def __init__(self):
        super().__init__()
        self.valid_masks = []
        self.probabilities = []

    def on_stage_start(self, state: State):
        print("Stage 3 started!")

    def on_batch_end(self, state: State):
        output = state.batch_out["logits"]
        input_masks = state.batch_in["targets"]
        for mask in input_masks:
            for m in mask:
                m = m.cpu().detach().numpy()
                if m.shape != (350, 525):
                    m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                self.valid_masks.append(m)

        for prob in output:
            for probability in prob:
                probability = probability.cpu().detach().numpy()
                if probability.shape != (350, 525):
                    probability = cv2.resize(
                        probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR
                    )
                self.probabilities.append(probability)

    def on_stage_end(self, state: State):
        class_params = {}
        for class_id in range(4):
            print(class_id)
            attempts = []
            for t in range(0, 100, 10):
                t /= 100
                for ms in [
                    0,
                    1000,
                    5000,
                    10000,
                    11000,
                    14000,
                    15000,
                    16000,
                    18000,
                    19000,
                    20000,
                    21000,
                    23000,
                    25000,
                    27000,
                ]:
                    masks = []
                    for i in range(class_id, len(self.probabilities), 4):
                        probability = self.probabilities[i]
                        predict, num_predict = post_process(sigmoid(probability), t, ms)
                        masks.append(predict)

                    d = []
                    for i, j in zip(masks, self.valid_masks[class_id::4]):
                        d.append(single_dice_coef(y_pred_bin=i, y_true=j))

                    attempts.append((t, ms, np.mean(d)))

            attempts_df = pd.DataFrame(attempts, columns=["threshold", "size", "dice"])

            attempts_df = attempts_df.sort_values("dice", ascending=False)
            print(attempts_df.head())
            best_threshold = attempts_df["threshold"].values[0]
            best_size = attempts_df["size"].values[0]

            class_params[class_id] = (best_threshold, best_size)
            np.save("./logs/class_params.npy", class_params)


class CustomInferCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(CallbackOrder.External)
        print("Custom infer callback is initialized")
        self.path = kwargs.get("path", None)
        self.threshold = kwargs.get("threshold", None)
        self.min_size = kwargs.get("min_size", None)
        self.class_params = dict()
        self.encoded_pixels = [None for i in range(14792)]
        self.pred_distr = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0}
        self.image_id = 0
        self.tta = kwargs.get("tta", None)

    def on_stage_start(self, state: "State"):
        if self.tta is not None:
            # state.model = tta.SegmentationTTAWrapper(state.model, tta.aliases.d4_transform())
            state.model = tta.TTAWrapper(state.model, tta.d4_image2mask)
            print(f"tta model created! type={type(state.model)}")

    def on_batch_end(self, state: State):
        #  print(type(state.model))
        # if not isinstance(state.model, tta.wrappers.SegmentationTTAWrapper):
        #    print("Not instance of tta")
        #    exit()
        for prob in state.batch_out["logits"]:
            for probability in prob:
                probability = probability.detach().cpu().numpy()
                if probability.shape != (350, 525):
                    probability = cv2.resize(
                        probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR
                    )
                prediction, num_predict = post_process(
                    sigmoid(probability),
                    threshold=self.threshold,
                    min_size=self.min_size,
                )
                if num_predict == 0:
                    self.pred_distr[-1] += 1
                    self.encoded_pixels[self.image_id] = ""
                else:
                    self.pred_distr[self.image_id % 4] += 1
                    r = mask2rle(prediction)
                    self.encoded_pixels[self.image_id] = r
                self.image_id += 1

    def on_stage_end(self, state: State):
        np.save("./logs/pred_distr.npy", self.pred_distr)
        sub = pd.read_csv(f"{self.path}/sample_submission.csv")
        sub["EncodedPixels"] = self.encoded_pixels
        sub.to_csv(
            "submission.csv", columns=["Image_Label", "EncodedPixels"], index=False
        )
        print("Inference is finished")


class PseudoLabelsCallback(InferCallback):
    def __init__(self, **kwargs):
        super().__init__()
        print("Pseudolabels callback is initialized")
        self.data_path = kwargs.get("data_path", None)
        self.data_folder = kwargs.get("data_folder", None)
        self.sub_name = kwargs.get("sub_name", None)
        self.low_threshold = kwargs.get("low_threshold", None)
        self.high_threshold = kwargs.get("high_threshold", None)
        # TODO self.always_take = kwargs.get("always_take", None)
        self.good_pixel_threshold = kwargs.get("good_pixel_threshold", None)
        self.mask_size_threshold = kwargs.get("mask_size_threshold", None)
        self.activation_threshold = kwargs.get("activation_threshold", None)
        self.encoded_pixels_pl = []
        self.names_pl = []
        self.image_id = 0
        self.sub = pd.read_csv(os.path.join(self.data_path, self.sub_name))

    def on_batch_end(self, state: State):
        preds_write = [False, False, False, False]
        allow = True
        for prob in state.batch_out["logits"]:
            for probability in prob:
                probability = probability.detach().cpu().numpy()
                probability = sigmoid(probability)
                pseudo_label = np.copy(probability)
                ones_condition = (pseudo_label < self.low_threshold) | (
                    pseudo_label > self.high_threshold
                )

                pseudo_label[ones_condition] = 1
                pseudo_label[~ones_condition] = 0
                val = (
                    pseudo_label.sum()
                    * 100
                    / (pseudo_label.shape[0] * pseudo_label.shape[1])
                )
                if val < self.good_pixel_threshold:
                    allow = False

                preds_write[self.image_id % 4] = (
                    probability,
                    self.sub.iloc[self.image_id]["Image_Label"],
                )
                self.image_id += 1

                if self.image_id % 4 == 0:
                    if allow:
                        for probability_new, name in preds_write:
                            self.names_pl.append(name)
                            predict_pl, num_predict_pl = post_process(
                                probability_new,
                                self.activation_threshold,
                                self.mask_size_threshold,
                            )
                            if num_predict_pl == 0:
                                self.encoded_pixels_pl.append("")
                            else:
                                r_pl = mask2rle(predict_pl)
                                self.encoded_pixels_pl.append(r_pl)
                    allow = True

    def on_stage_end(self, state: State):
        print("Processing")
        df_pseudo_labels = pd.DataFrame(
            {"Image_Label": self.names_pl, "EncodedPixels": self.encoded_pixels_pl}
        )
        df_pseudo_labels.to_csv(
            os.path.join(
                self.data_path,
                f"train_pl_{self.low_threshold}_{self.high_threshold}_{self.good_pixel_threshold}_{self.mask_size_threshold}.csv",
            ),
            index=False,
        )
        print(
            f"Pseudo-labels generation has finished! {len(df_pseudo_labels) / 4} labels were created!"
        )


"""
class DiceLossCallback(Callback):
    def __init__(self, input_key: str = "targets", output_key: str = "logits", prefix: str = "fscore"):
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        super().__init__(CallbackOrder.Metric)

    def on_batch_end(self, state: State) -> None:
        outputs = state.batch_out[self.output_key].detach().cpu()
        inputs = state.batch_in[self.input_key].detach().cpu()
        state.batch_metrics[self.prefix + "_10"] = f_score(outputs, inputs, beta=1., eps=10, threshold=0.5,
                                                   activation='sigmoid')
        state.batch_metrics[self.prefix + "_1e7"] = f_score(outputs, inputs, beta=1., eps=1e-7, threshold=0.5,
                                                   activation='sigmoid')

"""


class CustomDiceCallback(MetricCallback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid",
    ):
        super().__init__(
            prefix="dice_kirill",
            metric_fn=mean_dice_coef,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation,
        )


class CheckpointLoader(Callback):
    def __init__(self, checkpoint_path):
        super().__init__(CallbackOrder.Internal)
        self.checkpoint_path = checkpoint_path

    def on_stage_start(self, state: State):
        print(f"Checkpoint {self.checkpoint_path} is being loaded!")
        checkpoint = utils.load_checkpoint(self.checkpoint_path)
        utils.unpack_checkpoint(checkpoint, model=state.model)
