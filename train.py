import argparse
import datetime
import gc
import json
import os
import warnings

import segmentation_models_pytorch as smp
import torch.nn as nn
from catalyst import utils
from catalyst.contrib.criterion import DiceLoss
from catalyst.dl import DiceCallback
from catalyst.dl.callbacks import EarlyStoppingCallback, OptimizerCallback, CriterionCallback, \
    AUCCallback, CriterionAggregatorCallback, MixupCallback
from catalyst.dl.runner import SupervisedRunner
from catalyst.utils import set_global_seed, prepare_cudnn
from pytorch_toolbelt.inference.tta import TTAWrapper, fliplr_image2mask
from torch.optim.lr_scheduler import ReduceLROnPlateau

from callbacks import CustomCheckpointCallback, CustomDiceCallback
from dataset import prepare_loaders
from inference import predict
from lovasz_losses import CustomLovaszLoss
from models import get_model
from optimizers import get_optimizer
from utils import get_optimal_postprocess, NumpyEncoder

warnings.filterwarnings("once")

if __name__ == '__main__':
    """
    Example of usage:
    >>> python train.py --chunk_size=10000 --n_jobs=10

    """

    parser = argparse.ArgumentParser(description="Train model for understanding_cloud_organization competition")
    parser.add_argument("--path", help="path to files", type=str, default='f:/clouds')
    # https://github.com/qubvel/segmentation_models.pytorch
    parser.add_argument("--encoder", help="u-net encoder", type=str, default='resnet50')
    parser.add_argument("--encoder_weights", help="pre-training dataset", type=str, default='imagenet')
    parser.add_argument("--DEVICE", help="device", type=str, default='CUDA')
    parser.add_argument("--scheduler", help="scheduler", type=str, default='ReduceLROnPlateau')
    parser.add_argument("--loss", help="loss", type=str, default='BCEDiceLoss')
    parser.add_argument("--logdir", help="logdir", type=str, default=None)
    parser.add_argument("--optimizer", help="optimizer", type=str, default='RAdam')
    parser.add_argument("--augmentation", help="augmentation", type=str, default='default')
    parser.add_argument("--model_type", help="model_type", type=str, default='segm')
    parser.add_argument("--segm_type", help="model_type", type=str, default='Unet')
    parser.add_argument("--task", help="class or segm", type=str, default='segmentation')
    parser.add_argument("--num_workers", help="num_workers", type=int, default=0)
    parser.add_argument("--bs", help="batch size", type=int, default=4)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--lr_e", help="learning rate for encoder", type=float, default=1e-3)
    parser.add_argument("--num_epochs", help="number of epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation", help="gradient_accumulation steps", type=int, default=None)
    parser.add_argument("--height", help="height", type=int, default=320)
    parser.add_argument("--width", help="width", type=int, default=640)
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    parser.add_argument("--optimize_postprocess", help="to optimize postprocess", type=bool, default=False)
    parser.add_argument("--train", help="train", type=bool, default=False)
    parser.add_argument("--make_prediction", help="to make prediction", type=bool, default=False)
    parser.add_argument("--preload", help="save processed data", type=bool, default=False)
    parser.add_argument("--separate_decoder", help="number of epochs", type=bool, default=False)
    parser.add_argument("--multigpu", help="use multi-gpu", type=bool, default=False)
    parser.add_argument("--lookahead", help="use lookahead", type=bool, default=False)
    parser.add_argument("--use_tta", help="tta", type=bool, default=False)
    parser.add_argument("--resume_inference", help="path from which weights will be uploaded", type=str, default=None)
    parser.add_argument("--valid_split", help="choose validation split strategy", type=str, default="stratify")
    parser.add_argument("--convex_hull", help="use of convex hull in prediction", type=bool, default=True)
    parser.add_argument("--fp16", help="use fp16", type=bool, default=False)
    parser.add_argument("--pl_df_path", help="path to df with pseudo labels", type=str, default=None)
    parser.add_argument("--train_folder", help="name of train folder", type=str, default="train_images")
    parser.add_argument("--train_df_path", help="name of train df", type=str, default=None)
    parser.add_argument("--resume_train", help="name of train weights", type=str, default=None)
    parser.add_argument("--patience", help="patience parameter", type=int, default=2)
    parser.add_argument("--loss_smooth", help="smooth parameter", type=float, default=1.)

    args = parser.parse_args()

    if args.task == 'classification':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    set_global_seed(args.seed)
    prepare_cudnn(deterministic=True)

    sub_name = f'{args.segm_type}_aug_{args.augmentation}_{args.encoder}_bs_{args.bs}_{str(datetime.datetime.now().date())}'

    print(f'submission_{sub_name}.csv')

    logdir = f"./logs/{sub_name}" if args.logdir is None else args.logdir

    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)
    loaders, valid_len = prepare_loaders(path=args.path, bs=args.bs,
                              num_workers=args.num_workers, preprocessing_fn=preprocessing_fn, preload=args.preload,
                              image_size=(args.height, args.width), augmentation=args.augmentation, task=args.task,
                              validation_strategy=args.valid_split,
                              pl_df_path=args.pl_df_path,
                              train_folder=args.train_folder,
                              train_df_path=args.train_df_path)

    test_loader = loaders['test']
    del loaders['test']

    model = get_model(model_type=args.segm_type, encoder=args.encoder, encoder_weights=args.encoder_weights,
                      activation=None, task=args.task)

    print(model.activation)
    optimizer = get_optimizer(optimizer=args.optimizer, lookahead=args.lookahead, model=model,
                              separate_decoder=args.separate_decoder, lr=args.lr, lr_e=args.lr_e)

    if args.scheduler == 'ReduceLROnPlateau':
        print(f"Patience = {args.patience}")
        scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=args.patience)
    else:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=3)

    if args.loss == 'BCEDiceLoss':
        print(f"Loss smooth is {args.loss_smooth}")
        criterion = smp.utils.losses.BCEDiceLoss(eps=args.loss_smooth)
    elif args.loss == 'BCEJaccardLoss':
        criterion = smp.utils.losses.BCEJaccardLoss(eps=args.loss_smooth)
    elif args.loss == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "Lovasz":
        criterion = CustomLovaszLoss()
    elif args.loss == "complex":
        criterion = {
            "dice": DiceLoss(),
            "bce": nn.BCEWithLogitsLoss()
        }
    else:
        criterion = smp.utils.losses.BCEDiceLoss(eps=1.)

    if args.multigpu:
        model = nn.DataParallel(model)

    if args.task == 'segmentation':
        callbacks = [CustomDiceCallback(), DiceCallback(eps=0, prefix="dice_0"), DiceCallback(eps=1., prefix="dice_1"),
                     DiceCallback(eps=10., prefix="dice_10"), DiceCallback(eps=100., prefix="dice_100"),
                     EarlyStoppingCallback(patience=5, min_delta=0.001),
                     CriterionCallback(), CustomCheckpointCallback()]
    elif args.task == 'classification':
        callbacks = [AUCCallback(class_names=['Fish', 'Flower', 'Gravel', 'Sugar'], num_classes=4),
                     EarlyStoppingCallback(patience=5, min_delta=0.001), CriterionCallback(),
                     CustomCheckpointCallback()
                     ]

    print(callbacks)

    if args.gradient_accumulation:
        callbacks.append(OptimizerCallback(accumulation_steps=args.gradient_accumulation))

    if args.loss == "complex":
        callbacks += [
            CriterionCallback(
                input_key="features",
                prefix="loss_dice",
                criterion_key="dice",
                multiplier=2.0
            ),
            CriterionCallback(
                input_key="features",
                prefix="loss_bce",
                criterion_key="bce"
            ),
            CriterionAggregatorCallback(
                prefix="loss",
                loss_keys=["loss_dice", "loss_bce"],
                loss_aggregate_fn="sum"
            )
        ]

    fp16_params = None
    print(args.fp16)
    if args.fp16:
        print("FP16 is used")
        fp16_params = dict(opt_level="O1")

    if args.resume_train is not None:
        print("-------------------")
        print(f"resume weights path = {args.resume_train}")
        print("-------------------")
        checkpoint = utils.load_checkpoint(args.resume_train)
        model.cuda()
        utils.unpack_checkpoint(checkpoint, model=model)

    if args.use_tta:
        print("TTA model created")
        #model = tta.SegmentationTTAWrapper(model, tta.aliases.flip_transform(), merge_mode='tsharpen')
        model = TTAWrapper(model, fliplr_image2mask)

    runner = SupervisedRunner()

    if args.train:
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            callbacks=callbacks,
            logdir=logdir,
            num_epochs=args.num_epochs,
            fp16=fp16_params,
            verbose=True
        )

        with open(f'{logdir}/args.txt', 'w') as f:
            for k, v in args.__dict__.items():
                f.write(f'{k}: {v}' + '\n')

    gc.collect()

    class_params = None

    weights_path = f'{logdir}/checkpoints/best.pth'
    if args.resume_inference is not None:
        print("resume_inference")
        weights_path = args.resume_inference

    del loaders['train']

    print("-------------------")
    print(f"Weights path = {weights_path}")
    print("-------------------")
    checkpoint = utils.load_checkpoint(weights_path)
    model.cuda()
    utils.unpack_checkpoint(checkpoint, model=model)
    runner = SupervisedRunner(model=model)

    if args.optimize_postprocess:
        class_params = get_optimal_postprocess(loaders=loaders, runner=runner, valid_len=valid_len)
        with open(f'{logdir}/class_params.json', 'w') as f:
            json.dump(class_params, f, cls=NumpyEncoder)

    del loaders['valid']

    if args.make_prediction:
        loaders['test'] = test_loader
        del test_loader
        if class_params is None:
            class_params = {0: (0.3, 23000), 1: (0.5, 15000), 2: (0.5, 11000), 3: (0.6, 16000)}
        predict(loaders=loaders, runner=runner, class_params=class_params, path=args.path,
                sub_name=sub_name, convex_hull=args.convex_hull)
