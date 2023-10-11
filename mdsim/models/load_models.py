import torch
import os
from mdsim.models.schnet import SchNetWrap
from mdsim.models.dimenet_plus_plus import DimeNetPlusPlusWrap
from mdsim.models.forcenet import ForceNet
from mdsim.models.gemnet.gemnet import GemNetT
from mdsim.common.registry import registry

def load_pretrained_model(model_type, path = None, ckpt_epoch = -1, device = "cpu", train = True):
    if train:
        cname = 'best_checkpoint.pt' if ckpt_epoch == -1 else f"checkpoint{ckpt_epoch}.pt"
        ckpt_and_config_path = os.path.join(path, "checkpoints", cname)
        config = torch.load(ckpt_and_config_path, map_location=torch.device("cpu"))["config"]
    else:
        #load the final checkpoint instead of the best one
        ckpt_and_config_path = os.path.join(path, "ckpt.pth") if \
                os.path.exists(os.path.join(path, "ckpt.pth")) else os.path.join(path, "checkpoints", "best_checkpoint.pt")
        config = torch.load(ckpt_and_config_path, map_location=torch.device("cpu"))["config"]
    
    #load model
    model = registry.get_model_class(model_type)(**config["model_attributes"]).to(device)

    #get checkpoint
    print(f'Loading model weights from {ckpt_and_config_path}')
    try:
        checkpoint = {k: v.to(device) for k,v in torch.load(ckpt_and_config_path, map_location = torch.device("cpu"))['model_state'].items()}
    except:
        checkpoint = {k: v.to(device) for k,v in torch.load(ckpt_and_config_path, map_location = torch.device("cpu"))['state_dict'].items()}
    #checkpoint =  torch.load(ckpt_path, map_location = device)["state_dict"]
    try:
        new_dict = {k[7:]: v for k, v in checkpoint.items()}
        model.load_state_dict(new_dict)
    except:
        model.load_state_dict(checkpoint)

        
    return model, config["model_attributes"] 




        
