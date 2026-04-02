import argparse
import os
from typing import Tuple, List

import numpy as np
import torch

from base import Use_original, Collaboration
from clip import clip
from datasets import initialize_data
from defaults import _C as cfg
from network import load_base_model
from utils import initialize_experiment, update_args


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fairerclip")
    parser.add_argument("--config-file", default="configs/debias_counteranimal.yaml",
        metavar="FILE", help="path to config file", type=str,)  # e.g., debias_waterbird_mode.yaml
    parser.add_argument("--lr", default=1e-7, type=float)
    parser.add_argument("--img_lr", default=1e-7, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--img_epochs", default=10, type=int)
    parser.add_argument("--iter", default=1, type=int)
    parser.add_argument("--bs", default=256, type=int)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--ck", default=0, type=int)
    parser.add_argument("--first_modal", default="txt", type=str, choices=["txt", "img"])
    parser.add_argument("--att_mode", default="ori", type=str)  # mse | bafa | ori | none
    parser.add_argument("--learn_mode", default="proj", type=str)  # proj | linear | vpt | lora
    parser.add_argument("--txt_learn_mode", default="linear", type=str)  # linear | proj
    parser.add_argument("--model", default="clip_ViTL14",
        type=str, help="Model id string, e.g. clip_RN50 / clip_ViTB32 / clip_ViTB16 / clip_ViTL14", )
    parser.add_argument("--target_layer", default=1, type=int)
    parser.add_argument("--use_txt_ck_path", default=None, type=str, help="txt checkpoint filename under ./ckpt")
    parser.add_argument("--use_att", default=1, type=int)
    parser.add_argument("--use_ls", default=1, type=int, help="use similarity loss Ls in PGD")

    # MODE was used but not defined → add it with safe choices
    parser.add_argument("--mode", default="iccv", type=str,
        choices=["iccv", "no_li", "prompt"], help="training/eval routine switch",)

    # YACS-style override list (optional)
    parser.add_argument("--opts", help="Modify config options using the command-line (YACS style)",
        default=None, nargs="+",)
    return parser.parse_args()


def build_text_descriptions(dataset_name: str, train_loader=None) -> List[str]:
    """Return zero-shot text prompts per dataset; extend as needed."""
    if dataset_name == "waterbirds":
        return ["This is a picture of a landbird.", "This is a picture of a waterbird."]
    if dataset_name == "celebA":
        return ["A photo of a celebrity with dark hair.", "A photo of a celebrity with blond hair."]
    if dataset_name == "counteranimal":
        # Get class names from dataset
        if train_loader is not None and hasattr(train_loader.dataset, 'class_names'):
            class_names = train_loader.dataset.class_names
            return [f"A photo of a {name}." for name in class_names]
        return []
    # Fallback: empty list (downstream code guards against empty)
    return []


def safe_load_state_dict(module: torch.nn.Module, ckpt_path: str) -> None:
    """Load a state dict safely on CPU/CUDA depending on availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(os.path.join("./ckpt", ckpt_path), map_location=device)
    module.load_state_dict(state)


def main():
    args = parse_args()

    # Load YAML config
    cfg.merge_from_file(args.config_file)

    # Merge a few runtime options into cfg (works with YACS-style cfg as well)
    cfg.merge_from_list(["lr", args.lr, "load_base_model", args.model, "att_mode",  args.att_mode, "learn_mode",
                         args.learn_mode, "embeddings_dir", "new_test", "txt_learn_mode", args.txt_learn_mode,
            "target_layer", args.target_layer, "arch", args.model.split("_")[-1],  # e.g., 'RN50' from 'clip_RN50'
            "use_ls", bool(args.use_ls)
        ])
    cfg.img_lr, cfg.num_workers = args.img_lr, 2

    # Optional extra overrides from CLI
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Let project utilities sync args ↔ cfg if they do so
    update_args(args, cfg)

    # Seeding
    cfg.seed = args.seed
    seed_everything(cfg.seed)

    # ----------------------------------------------------------
    # Load base model + transforms + helper fns
    # ----------------------------------------------------------
    base_model_args = cfg.load_base_model.split("_")
    (base_model, base_transform, get_embeddings, get_dataset_embeddings, get_zeroshot_predictions,
    ) = load_base_model(base_model_args, cfg, clip=clip)

    # Wrap into project helper
    uo = Use_original(base_model, base_transform, get_embeddings, get_dataset_embeddings, get_zeroshot_predictions, cfg)

    # I/O prep
    os.makedirs(cfg.embeddings_dir, exist_ok=True)

    # Data: initialize_data returns a loader factory; call it to get splits
    load_dataloaders = initialize_data(args)
    train_loader_base, val_loader_base, test_loader_base = load_dataloaders(
        args, train_shuffle=False, transform=base_transform)

    # Build text prompts (after loading data for counteranimal)
    text_descriptions = build_text_descriptions(getattr(cfg, "dataset", ""), train_loader_base)

    # Experiment bookkeeping (dirs, loggers, etc.)
    initialize_experiment(cfg)

    # Model collaboration wrapper
    model = Collaboration(cfg, uo, train_loader_base, val_loader_base, test_loader_base)

    # Precompute CLIP image features if requested
    if args.iter > 0:
        model.exper_set(False, cfg.att_mode, cfg.learn_mode, text_descriptions)
        model.clip_img_feat(data_t=cfg.dataset)
    else:
        model.clip_img_feat(data_t=cfg.dataset)


    model.img_model.train(),  model.txt_model.train()
    # Training / alternating routine
    t_type = args.first_modal  # 'txt' or 'img'
    for i in range(max(args.iter, 0)):
        print("===================", "iter :", i, " : ", t_type, "times", "===========================")
        if args.mode == "iccv":
            if t_type == "txt":
                # text step
                model.img_model.eval()
                if args.use_txt_ck_path is None:
                    model.text_iccv(args.epochs, i, use_att=args.use_att)
                else:
                    safe_load_state_dict(model.txt_model, args.use_txt_ck_path)
                    model.test()
                t_type = "img"
            else:
                # image step
                model.img_iccv(args.img_epochs, i)
                t_type = "txt"

        elif args.mode == "no_li":
            if t_type == "txt":
                model.text_iccv_test(args.epochs)
                t_type = "img"
            else:
                t_type = "txt"

        elif args.mode == "prompt":
            if t_type == "txt":
                
                model.text_iccv_test(args.epochs, args.mode)
                t_type = "img"
            else:
                t_type = "txt"
        elif args.mode == "orth-cali":
            P = model.P.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).half()
            model.test(P)
            pass


    if hasattr(model, "test"):
        model.test()
        model.cifar_test2()
        model.imagenet_test()


if __name__ == "__main__":
    main()
