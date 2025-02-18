import config
import torch

class bert_dataset:
    def __init__(self,review,target):
        self.review =review 
        self.target = target 
        self.tokenizer = config.tokenizer
        self.max_len  = config.max_len
    def __getitem__(self,item):
        review = str(self.review[item])
        review = " ".join(review.split())
    
        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens = True,
            max_length = self.max_len
                   )

        ids  = inputs["input_ids"]
        mask = inputs["special_tokens_mask"]
        token_type_ids =  inputs["token_type_ids"]

        padding_len = self.max_len - len(ids)

        ids = ids +([0] *padding_len)
        mask = mask +([0] *padding_len)
        token_type_ids = token_type_ids +([0] *padding_len)       

        return {
            'ids': torch.tensor(ids,dtype = torch.long),
            'mask' : torch.tensor(mask , dtype = torch.long), 
            'token_type_ids': torch.tensor(token_type_ids,dtype = torch.long),
            'targets': torch.tensor(self.target[item] ,dtype=torch.float)

        }  

    def __len__(self):
        return len(self.review)