from tqdm import tqdm
import torch 
import torch.nn as nn 

def train_fn(data_loader,model ,optimizer,device,scheduler):
    model.train()

    for i, d in tqdm(enumerate(data_loader),total = len(data_loader)):
         ids = d["ids"]
         mask =d["mask"]
         token_type_ids =d["token_type_ids"]
         targets = d["targets"]      

         ids = ids.to(device,dtype = torch.long)
         token_type_ids= token_type_ids.to(device,dtype=torch.long)
         mask = mask.to(device,dtype = torch.long)
         targets = targets.to(device,dtype=torch.float)
         model = model.to(device)

         optimizer.zero_grad()
         outputs = model(
             ids=ids , 
             mask=mask ,
             token_type_ids=token_type_ids
         )

         loss =  nn.BCEWithLogitsLoss(outputs,targets)
         loss.backward()
         optimizer.step()
         scheduler.step()


def eval_fn(data_loader,model,device):
    model.eval()
    in_targets= []
    in_outputs= []
    with torch.no_grad():
        for i, d in tqdm(enumerate(data_loader),total = len(data_loader)):
                ids = d["ids"]
                mask =d["mask"]
                token_type_ids =d["token_type_ids"]
                targets = d["targets"]      

                ids = ids.to(device,dtype = torch.long)
                token_type_ids= token_type_ids.to(device,dtype=torch.long)
                mask = mask.to(device,dtype = torch.long)
                targets = targets.to(device,dtype=torch.float)
                model = model.to(device)

             
                outputs = model(
                    ids=ids , 
                    mask=mask ,
                    token_type_ids=token_type_ids
                )

                in_targets.extend(targets.cpu().detach().numpy().tolist())
                in_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
               
    return in_outputs,in_targets
            

            



