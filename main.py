import argparse
import torch
import datetime
import json
import yaml
import os

from dataloader import get_dataloader_s1, get_dataloader_s2
from model.user_model import VAE
from utils import train_VAE, evaluate
from model.imputation_model import UDMI

parser = argparse.ArgumentParser(description="User Behavior-aware Imputation")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cpu')
parser.add_argument("--save", type=str, default="folder for saving")
parser.add_argument("--stage", type=str, default="1",
                    help="1: behavior extraction, 2: imputation")
parser.add_argument("--if_load", type=str, default="0")
parser.add_argument("--l_vae_path", type=str, default="")
parser.add_argument("--i_vae_path", type=str, default="")
parser.add_argument("--dm_path", type=str, default="")

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)
print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
foldername = (
    str(args.save) + "/" + current_time + "/"
)
print('model folder:', foldername)

os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

if args.stage == "1": # behavior extraction
    # train instance
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader_s1(config, int(args.stage), mode="train")
    i_vae = VAE(config, args.device).to(args.device)
    if args.if_load == "0":
        train_VAE(i_vae, config, train_loader, valid_loader, foldername)
    else:
        i_vae.load_state_dict(torch.load(args.i_vae_path))

    # train instance
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader_s1(config, int(args.stage), subsequence=True,
                                                                                     mode="train")
    l_vae = VAE(config, args.device).to(args.device)
    if args.if_load == "0":
        train_VAE(i_vae, config, train_loader, valid_loader, foldername)
    else:
        l_vae.load_state_dict(torch.load(args.l_vae_path))
elif args.stage == "2": # imputation
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader_s2(config, int(args.stage), mode="train")
    imputation_model = UDMI(config, args.device).to(args.device)
    evaluate(imputation_model, config, train_loader, valid_loader, test_loader, scaler, mean_scaler, foldername, args.dm_path)
