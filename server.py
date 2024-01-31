from inferlight import LightWrapper, BaseInferLightWorker
import time
# from sanic import Sanic
# from sanic.response import json as json_response
from fastapi import FastAPI,Request
import uvicorn
import random
import logging

from transformers import BertTokenizer, BertForSequenceClassification

import numpy as np
from scipy.special import softmax
import torch
from torch import nn
import multiprocessing as mp


logging.basicConfig(level=logging.INFO)

class BertModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.bert.eval()
        self.device = torch.device('cuda' if config.get('use_cuda') else 'cpu')
        self.bert.to(self.device)

    def forward(self, inputs):
        return self.bert(**inputs).logits

class MyWorker(BaseInferLightWorker):

    def load_model(self, model_args):
        self.model = BertModel(model_args)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if model_args.get('use_cuda') else 'cpu')

    def build_batch(self, requests):
        # encoded_input type = {}
        # input_ids, token_type_ids, attention_mask
        encoded_input = self.tokenizer.batch_encode_plus(requests, 
                                                         return_tensors='pt',
                                                         padding=True,
                                                         truncation=True,
                                                         max_length=512)
        print("encoded_input.shape = {}".format(encoded_input["input_ids"].shape))
        return encoded_input.to(self.device)

    @torch.no_grad()
    def inference(self, batch):
        model_output = self.model.forward(batch).cpu().numpy()
        scores = softmax(model_output, axis=1)
        ret = [x.tolist() for x in scores]
        return ret
        

if __name__=='__main__':
    config = {
        'model':"bert-base-uncased",
        'use_cuda':True
    }

    mp.set_start_method("spawn", force= True)
    torch.multiprocessing.set_start_method("spawn", force=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    bert_model = bert_model.to(device)
    bert_model.eval()

    wrapped_model = LightWrapper(MyWorker, config, batch_size=4, max_delay=0.05)

    app = FastAPI()

    @torch.no_grad()
    @app.post("/predict")
    async def predict(input_data: Request):
        data = await input_data.json()
        text = data["text"]
        inputs = tokenizer(text, return_tensors='pt')
        inputs = inputs.to(device)
        outputs = bert_model(**inputs)
        predicted_class = outputs.logits.argmax().item()
        print("predicted_class = {}".format(predicted_class))
        return {"predicted_class": predicted_class}
    

    @app.post('/batch_predict')
    async def batched_predict(input_data: Request):
        data = await input_data.json()
        dummy_input = data["text"]
        response = await wrapped_model.predict(dummy_input)
        if not response.succeed():
            return {'output':None, 'status':'failed'}
        # result = {'input':str(dummy_input),'output': response.result}
        result = {"status":"success"}
        return result

    uvicorn.run(app, host="0.0.0.0", port=8000)
