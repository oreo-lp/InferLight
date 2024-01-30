from fastapi import FastAPI, Request
import uvicorn
import asyncio
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model = model.to(device)
model.eval()

@torch.no_grad()
@app.post("/predict")
async def predict(input_data: Request):
    data = await input_data.json()
    text = data["text"]
    inputs = tokenizer(text, return_tensors='pt')
    inputs = inputs.to(device)
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax().item()
    print("predicted_class = {}".format(predicted_class))
    return {"predicted_class": predicted_class}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)