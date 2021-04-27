from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--port", "-p", type=int, default=None)
parser.add_argument("--ngrok", "-n", action="store_true")
parser.add_argument("--model",
                    "-m",
                    type=str,
                    default="microsoft/DialoGPT-medium")
params = parser.parse_args()

app = Flask(__name__)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained(params.model)
model = AutoModelForCausalLM.from_pretrained(params.model).to(device)
step = 0


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    global step
    global chat_history_ids

    userText = request.args.get('msg')

    if "bye" in userText.lower():
        step = 0
        return "BYE!!!"

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(userText + tokenizer.eos_token,
                                          return_tensors='pt').to(device)

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids],
                              dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(bot_input_ids,
                                      max_length=1000,
                                      pad_token_id=tokenizer.eos_token_id)

    step += 1
    response = tokenizer.decode(chat_history_ids[:,
                                                 bot_input_ids.shape[-1]:][0],
                                skip_special_tokens=True)

    return str(response)


if __name__ == "__main__":
    if params.ngrok:
        run_with_ngrok(app)
        app.run()
    else:
        app.run(port=params.port)
