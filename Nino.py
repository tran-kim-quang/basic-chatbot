import random
import json
import torch
from Nino_model import NeuralNet
from nltk_utils import tokenize, bag_word

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

file = "Nino_data.pth"
data = torch.load(file)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
words = data['words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

name = "Nino"

def get_chat(msg):
    boss = tokenize(msg)
    X = bag_word(boss, words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, pred = torch.max(output, dim=1)

    tag = tags[pred.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][pred.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "I do not understand about it..."


# print("What should i call you?")
# user = input("You can call me: ")
# print(f"Hi {user}! Wanna share something with me? Feel free if you want to leave the conversation just type 'quit' ^_^")
# while True:
#     boss = input(f"{user}: ")
#     if boss == 'quit':
#         break
#     boss = tokenize(boss)
#     X = bag_word(boss, words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, pred = torch.max(output, dim=1)

#     tag = tags[pred.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][pred.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 print(f"{name}: {random.choice(intent['responses'])}")
#     else:
#         print(f"{name}: I do not understand about it...")