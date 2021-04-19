from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)  #starts ngrok when the app is run

from transformers import pipeline, Conversation

conversational_pipeline = pipeline("conversational",
                                   model="microsoft/DialoGPT-medium")

user_str = None
conv = None
cpipe = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')

    if conv is None:
        conv = Conversation(userText)
    else:
        conv.add_user_input(userText)
    cpipe = conversational_pipeline([conv], device=1)
    response = cpipe.generated_responses[-1]
    return str(response)


if __name__ == "__main__":
    app.run()
