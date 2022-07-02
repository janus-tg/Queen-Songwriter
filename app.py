from flask import Flask, render_template, request
import torch


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def formInput():
    text = request.form['text']
    print(text)
    return render_template('index.html', oldInput = text, newLyrics = text + "resultwefn urwibhiwufw euiore2or oeirhe20or oehri2oebr oeirhoeb oierhoerbh oehroie2hro oeiufewo")

def generate(model, tokenizer, prompt, entryCount=1, entryLength=75, topP=0.8, temp=1.0):

    model.eval()
    generatedList = []
    filterVal = -float("Inf")

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

                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generatedList.append(output_text)
                    break
            
            if not entryFin:
              output_list = list(generated.squeeze().numpy())
              output_text = f"{tokenizer.decode(output_list)}" 
              generatedList.append(output_text)
                
    return generatedList[0]
if __name__ == "__main__":
    app.run(debug=True)