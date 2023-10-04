import torch
import os
from mdsim.models.schnet import SchNetWrap

def load_schnet_model(path = None, ckpt_epoch = -1, num_interactions = None, device = "cpu", from_pretrained=True, train = True):
    
    if train:
        cname = 'best_checkpoint.pt' if ckpt_epoch == -1 else f"checkpoint{ckpt_epoch}.pt"
        ckpt_and_config_path = os.path.join(path, "checkpoints", cname)
        schnet_config = torch.load(ckpt_and_config_path, map_location=torch.device("cpu"))["config"]
    else:
        #load the final checkpoint instead of the best one
        ckpt_and_config_path = os.path.join(path, "ckpt.pth") if \
                os.path.exists(os.path.join(path, "ckpt.pth")) else os.path.join(path, "checkpoints", "best_checkpoint.pt")
        #temp hardcoded path since we haven't been saving the model config
        schnet_config = torch.load('/pscratch/sd/s/sanjeevr/MODELPATH/schnet/md17-ethanol_1k_schnet/checkpoints/best_checkpoint.pt', \
                            map_location=torch.device("cpu"))["config"]
    
    if num_interactions: #manual override
        schnet_config["model_attributes"]["num_interactions"] = num_interactions
    # keep = list(schnet_config["model_attributes"].keys())
    # args = {k: schnet_config["model_attributes"][k] for k in keep}
    model = SchNetWrap(**schnet_config["model_attributes"]).to(device)

    if from_pretrained:
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

        
    return model, schnet_config["model_attributes"] 




        
