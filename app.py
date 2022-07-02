from flask import Flask, render_template, request
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from tqdm import tqdm, trange

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def formInput():
    text = request.form['text']
    modelOutput = generate(model, tokenizer, text)
    return render_template('index.html', oldInput = text, newLyrics = modelOutput)

def generate(model, tokenizer, prompt, entryCount=1, entryLength=75, topP=0.8, temp=1.0):

    model.eval()
    generatedList = []
    filterVal = -float("Inf")
    generatedNum = 0

    with torch.no_grad():

        for entry_idx in trange(entryCount):

            entryFin = False
            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entryLength):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temp if temp > 0 else 1.0)

                sortedLogits, sortedIndices = torch.sort(logits, descending=True)
                cumulativeProbs = torch.cumsum(torch.nn.functional.softmax(sortedLogits, dim=-1), dim=-1)

                sortedIndicesToRemove = cumulativeProbs > topP
                sortedIndicesToRemove[..., 1:] = sortedIndicesToRemove[
                    ..., :-1
                ].clone()
                sortedIndicesToRemove[..., 0] = 0

                indices_to_remove = sortedIndices[sortedIndicesToRemove]
                logits[:, indices_to_remove] = filterVal

                next_token = torch.multinomial(torch.nn.functional.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|endoftext|>"):
                    entryFin = True

                if entryFin:

                    generatedNum += 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generatedList.append(output_text)
                    break
            
            if not entryFin:
              output_list = list(generated.squeeze().numpy())
              output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 
              generatedList.append(output_text)
                
    return generatedList[0]

if __name__ == "__main__":
    global model; global tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.load_state_dict(torch.load('modelWeights.pt', map_location=torch.device('cpu')))
    app.run(debug=True)