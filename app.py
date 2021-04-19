from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import argparse
from transformers import pipeline, Conversation

parser = argparse.ArgumentParser()
parser.add_argument("--port", "-p", type=int, default=None)
parser.add_argument("--ngrok", "-n", action="store_true")
parser.add_argument("--gpt_capacity", "-c", type=str, default="medium")

params = parser.parse_args()

app = Flask(__name__)
# run_with_ngrok(app)  #starts ngrok when the app is run

conversational_pipeline = pipeline(
    "conversational", model=f"microsoft/DialoGPT-{params.gpt_capacity}")

conv = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    global conv
    global conversational_pipeline

    userText = request.args.get('msg')

    if conv is None:
        conv = Conversation(userText)
    else:
        conv.add_user_input(userText)

    if "bye" in userText.lower():
        conv = None
        return ("bye\n" + ("-" * 30 + "\n") * 3)

    cpipe = conversational_pipeline([conv], device=1)
    response = cpipe.generated_responses[-1]
    return str(response)


if __name__ == "__main__":
    if params.ngrok:
        run_with_ngrok(app)
        app.run()
    else:
        app.run(port=params.port)
