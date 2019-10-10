import argparse
import datetime
import gc
import json
import os
import warnings

# from catalyst.contrib.criterion.lovasz import LovaszLossMultiClass, LovaszLossBinary
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import ttach as tta
from catalyst import utils
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, OptimizerCallback, CriterionCallback, AUCCallback
from catalyst.dl.runner import SupervisedRunner
from catalyst.utils import set_global_seed, prepare_cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import prepare_loaders
from inference import predict
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
    parser.add_argument("--lr_e", help="learning rate for decoder", type=float, default=1e-3)
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

    args = parser.parse_args()

    if args.task == 'classification':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    set_global_seed(args.seed)
    prepare_cudnn(deterministic=True)

    sub_name = f'Model_{args.task}_{args.model_type}_{args.encoder}_bs_{args.bs}_{str(datetime.datetime.now().date())}'
    logdir = f"./logs/{sub_name}" if args.logdir is None else args.logdir

    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)
    loaders = prepare_loaders(path=args.path, bs=args.bs,
                              num_workers=args.num_workers, preprocessing_fn=preprocessing_fn, preload=args.preload,
                              image_size=(args.height, args.width), augmentation=args.augmentation, task=args.task)
    test_loader = loaders['test']
    del loaders['test']

    model = get_model(model_type=args.segm_type, encoder=args.encoder, encoder_weights=args.encoder_weights,
                      activation=None, task=args.task)

    optimizer = get_optimizer(optimizer=args.optimizer, lookahead=args.lookahead, model=model,
                              separate_decoder=args.separate_decoder, lr=args.lr, lr_e=args.lr_e)

    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, factor=0.6, patience=2)
    else:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.3, patience=3)

    if args.loss == 'BCEDiceLoss':
        criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    elif args.loss == 'BCEJaccardLoss':
        criterion = smp.utils.losses.BCEJaccardLoss(eps=1.)
    elif args.loss == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = smp.utils.losses.BCEDiceLoss(eps=1.)

    if args.multigpu:
        model = nn.DataParallel(model)

    if args.task == 'segmentation':
        callbacks = [DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001), CriterionCallback()]
    elif args.task == 'classification':
        callbacks = [AUCCallback(class_names=['Fish', 'Flower', 'Gravel', 'Sugar'], num_classes=4), EarlyStoppingCallback(patience=5, min_delta=0.001), CriterionCallback()]

    if args.gradient_accumulation:
        callbacks.append(OptimizerCallback(accumulation_steps=args.gradient_accumulation))

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
            verbose=True
        )

        with open(f'{logdir}/args.txt', 'w') as f:
            for k, v in args.__dict__.items():
                f.write(f'{k}: {v}' + '\n')

    torch.cuda.empty_cache()
    gc.collect()

    class_params = None

    if args.optimize_postprocess:
        del loaders['train']
        weights_path = f'{logdir}/checkpoints/best.pth'
        if args.resume_inference is not None:
            print("resume_inference")
            weights_path = args.resume_inference
        checkpoint = utils.load_checkpoint(weights_path)
        model.cuda()
        utils.unpack_checkpoint(checkpoint, model=model)
        runner = SupervisedRunner(model=model)
        if args.resume_inference:
            class_params = get_optimal_postprocess(loaders=loaders, runner=runner, logdir=logdir, resume_inference_path=weights_path)
        else:
            class_params = get_optimal_postprocess(loaders=loaders, runner=runner, logdir=logdir)
        with open(f'{logdir}/class_params.json', 'w') as f:
            json.dump(class_params, f, cls=NumpyEncoder)

    torch.cuda.empty_cache()
    gc.collect()

    if args.make_prediction:
        loaders['test'] = test_loader
        weights_path = f'{logdir}/checkpoints/best.pth'
        if args.resume_inference is not None:
            print("resume_inference")
            weights_path = args.resume_inference
        checkpoint = utils.load_checkpoint(weights_path)
        model.cuda()
        utils.unpack_checkpoint(checkpoint, model=model)
        runner = SupervisedRunner(model=model)
        if not class_params:
            if not args.resume_inference:
                with open(f'{logdir}/class_params.json', 'r') as f:
                    class_params = json.load(f)
            else:
                class_params = {0: (0.2, 30000), 1: (0.7, 11000), 2: (0.6, 30000), 3: (0.3, 16000)}

        if args.use_tta:
            print("TTA started")
            tta_model = tta.SegmentationTTAWrapper(runner.model, tta.aliases.d4_transform(), merge_mode='mean')
            tta_runner = SupervisedRunner(
                model=tta_model
            )
            predict(loaders=loaders, runner=tta_runner, class_params=class_params, path=args.path, sub_name=sub_name)
         #   predict(loaders=loaders, runner=runner, class_params=class_params, path=args.path, sub_name=sub_name)
        else:
            predict(loaders=loaders, runner=runner, class_params=class_params, path=args.path, sub_name=sub_name)
