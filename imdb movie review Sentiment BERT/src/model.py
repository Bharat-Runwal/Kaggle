import transformers
import torch.nn as nn 

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased,self).__init__() 
        self.bert = transformers.BertModel('bert-base-uncased')
        self.b_dropout = nn.Dropout(0.3)
        self.out = nn.Linear(768,1)

    def forward(self,ids,mask,token_type_ids):
        _,out2  = self.bert(
            ids,attention_mask = mask, token_type_ids=token_type_ids
        )
        b_out = self.b_dropout(out2)
        output = self.out(b_out)
        return output

    
        
