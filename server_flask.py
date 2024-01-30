from flask import Flask, request, jsonify
import json
import torch

from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model = model.to(device)
model.eval()

@torch.no_grad()
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_data(as_text = True)
    data = json.loads(data)
    text = data["text"]
    inputs = tokenizer(text, return_tensors='pt')
    inputs = inputs.to(device)
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax().item()
    print("predicted_class = {}".format(predicted_class))
    return {"predicted_class": predicted_class}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)