# Chat Bot
The purpose of this project is to creat a chatbot for the Capstone project of Udacity's Machine Learning Engineer program.

The implementation is straightforward with a Feed Forward Neural net with 2 hidden layers.

You need to install PyTorch and nltk.

## Usage
To **train** the ML model, type in the terminal:
```console
python train.py
```
To **test** the ML model, type in the terminal:
```console
python chat.py
```
or you can use test.ipynb

To **execute** the chatbot application, type in the terminal:
```console
python app.py
```
## Customize
You can customize intents documents according to your own use case. You need to define a new `tag`, possible `patterns`, and possible `responses` for the chat bot. 

```console
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi","Hey","How are you","Is anyone there?","Hello","Good day"],
      "responses": ["Hey :-)","Hello, thanks for visiting","Hi there, what can I do for you?","Hi there, how can I help?"]
    },
    ...
  ]
}
```
