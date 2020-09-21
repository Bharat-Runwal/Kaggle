import transformers

max_len = 512 
train_batch = 8
valid_batch =4 
epochs = 10
accumulation =2
model_path = "model.bin"
# bert_path = "C:/Users/lenovo/Desktop/Kaggle/imdb movie review Sentiment BERT/input/bert_base_uncased/"
training_file = "C:/Users/lenovo/Desktop/Kaggle/imdb movie review Sentiment BERT/input/imdb.csv"
# model = AutoModelWithLMHead.from_pretrained("bert-base-uncased")
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

 