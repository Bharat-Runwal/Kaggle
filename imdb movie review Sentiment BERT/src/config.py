import transformers

max_len = 512 
train_batch = 8
valid_batch =4 
epochs = 10
accumulation =2
model_path = "model.bin"
training_file = "../input/imdb.csv"
# model = AutoModelWithLMHead.from_pretrained("bert-base-uncased")
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

 