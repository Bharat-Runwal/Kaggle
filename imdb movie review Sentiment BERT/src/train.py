import torch 
import config 
import dataset 
import engine 
import pandas as pd 
import numpy as np 
from sklearn import model_selection,metrics
from model import BERTBaseUncased
from transformers import AdamW
from transformers import WarmupLinearSchedule


def run():
    df = pd.read_csv(config.training_file).fillna("none")
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)

    df_train,df_valid= model_selection.train_test_split(df, test_size=0.1,random_state =42 , stratify= df.sentiment.values)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    

    train_dataset =dataset.bert_dataset(review = df_train.review.values, target=df_train.sentiment.values)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch,
        num_workers=4
    )


    
    valid_dataset =dataset.bert_dataset(review = df_valid.review.values, target=df_valid.sentiment.values)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.valid_batch,
        num_workers=1
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = BERTBaseUncased()

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias","LayerNorm.bias","LayerNorm.weight"]
    optimizer_params= [
        {"params": [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.001},
          {"params": [p for n,p in param_optimizer if  any(nd in n for nd in no_decay)],'weight_decay':0.0}
    ]

    num_train_steps = int(len(df_train)/config.train_batch) * config.epochs

    optimizer = AdamW(optimizer_params,lr=3e-5)
     
    scheduler= WarmupLinearSchedule(
        optimizer=optimizer,
         warmup_steps=0,
         t_total=num_train_steps
         )
    
    best_accuracy = 0 
    for epoch in range(config.epochs):
        engine.train_fn(data_loader=train_data_loader,model=model,
        optimizer=optimizer,device=device,scheduler=scheduler)

        outputs,targets = engine.eval_fn(valid_data_loader,model=model,device=device)
        outputs =  np.array(outputs)>=0.5
        accuracy = metrics.accuracy_score(outputs,targets)
        print(f"Accuracy Score= {accuracy}")
        if accuracy > best_accuracy :
            torch.save(model.state_dict(),config.model_path)
            best_accuracy=accuracy
            


if __name__ == "__main__":
    run()


    