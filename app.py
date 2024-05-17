import gradio as gr
from huggingface_hub import InferenceClient
import gdown
import os
import torch
import re
from transformers import AutoModel, BertTokenizerFast, TFBertModel, AutoTokenizer, BertTokenizer, AdamW
import torch.nn as nn

###############
# Load model
###############
url='https://drive.google.com/file/d/1cT8q8-rU-5T7TEKAB5oGkIAhS6dOD3O9/view?usp=sharing'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
model_file_name = 'best_model.pt'

if not os.path.exists(model_file_name):
    gdown.download(url, model_file_name, quiet=False)
else:
    print('Model already downloaded')

###############
# Preprocess helpers
###############

def preprocess_text(text):
    # Transform URLs
    text = re.sub(r'http\S+', '<URL>', text)

    # Transform other user mentions
    text = re.sub(r'@\w+', '<USUARIO>', text)

    # Delete emojis
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+', '', text)
    text = re.sub(r':\)|;\)|:-\)|:-\(|:-\(|:-\*|:-\)|:P|:D|:<|:-P', '<EMOJI>', text)

    # Transform hashtags
    text = re.sub(r'#\w+', '<HASH_TAG>', text)

    return text

MAX_LENGHT = 25
tokenizer = BertTokenizerFast.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')

def tokenize(data_to_tokenize):
  return tokenizer.batch_encode_plus(
    data_to_tokenize,
    max_length = MAX_LENGHT,
    pad_to_max_length=True,
    truncation=True
  )

###############
# Prediction
###############

def prediction(text):
    text = preprocess_text(text)
    tokenized_text = tokenize([text])

    result = 0
    result_to_text = 'Falsa' if result == 0 else 'Verdadera'
    return f"La noticia es {result_to_text}"

###############
# Deploy
###############

iface = gr.Interface(
    fn=prediction,
    inputs="text",
    outputs="label",
    title="Detector de tweets de noticias falsas",
    description="Introduce el texto de una noticia publicada en un tweet y descubre si es verdadera o falsa",
)

if __name__ == "__main__":
    iface.launch()