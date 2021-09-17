from pyfirmata.pyfirmata import Board
from pyfirmata import OUTPUT, Arduino, util
from wikipedia import exceptions
from task import wishme
import datetime
from task import inputfuction
from task import noninputfunction
from speak import say
from listen import listen
import random
import json
import torch
from Brain import NeuralNet
from Neuralnetwork import bag_of_words, tokenzie
import time
import pywhatkit as kit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("intents.json", 'r') as json_data:
    intents = json.load(json_data)

FILE = "Training.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

##########################################################################################################################
#                                                        SKYNET                                                       #
##########################################################################################################################
Name = "skynet"


# arduino ke liye h bhai ye
# board = Arduino('COM3')
# board.digital[12].mode = OUTPUT
# board.digital[11].mode = OUTPUT
# board.digital[10].mode = OUTPUT
# board.digital[9].mode = OUTPUT
# board.digital[12].write(1)
# board.digital[11].write(1)
# board.digital[10].write(1)
# board.digital[9].write(1)

wishme()


def main():
    kaam = listen()
    sentence = listen()
    result = str(sentence)

    if sentence == "bye":
        say("take care arjun sir. byeeee")
        exit()

    sentence = tokenzie(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)

    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                reply = random.choice(intent["responses"])
                if "time" in reply:
                    noninputfunction(reply)
                elif "date" in reply:
                    noninputfunction(reply)
                elif "day" in reply:
                    noninputfunction(reply)
                elif "wikipedia" in reply:
                    inputfuction(reply, result)
                elif "google" in reply:
                    inputfuction(reply, result)
                elif "screenshot" in reply:
                    noninputfunction(reply)
                elif "youtube" in kaam:
                    kaam = kaam.replace("youtube", "")
                    kaam = kaam.replace("skynet", "")
                    kaam = kaam.replace("kar do", "")
                    kaam = kaam.replace("do", "")
                    kaam = kaam.replace("please", "")
                    kaam = kaam.replace("all", "")
                    kaam = kaam.replace("play", "")
                    kaam = kaam.replace("sing", "")
                    kaam = kaam.replace("hello world", "")
                    say("opening youtube")
                    kit.playonyt(listen())
                # elif "light on" in kaam:
                #     say("ok sir, i am turning on the light")
                #     board.digital[12].write(0)
                # elif "light off" in kaam:
                #     say("ok sir, i am turning off the light")
                #     board.digital[12].write(1)
                # elif "relay one on" in kaam:
                #     say("ok sir, i am turning on the relay 2")
                #     board.digital[11].write(0)
                # elif "relay one off" in kaam:
                #     say("ok sir, i am turning off the relay 2")
                #     board.digital[11].write(1)
                # elif "relay 2 on" in kaam:
                #     say("ok sir, i am turning on the relay 3")
                #     board.digital[10].write(0)
                # elif "relay 2 off" in kaam:
                #     say("ok sir, i am turning on the relay 3")
                #     board.digital[10].write(1)

                # elif "screenshot" in kaam:
                #     try:
                #         say("sir, please tell me the name for this screenshot file")
                #         name = listen().lower()
                #         say("please sir hold the screen for few seconds, I am taking screenshot")
                #         time.sleep(2)
                #         img = pyautogui.screenshot()
                #         img.save(f"{name}.png")
                #         say("ok done sir, The screenshot is saved in our main folder.")
                #     except Exception as e:
                #         say("sorry sir, but space is not define.")
                else: 
                        say(reply)

while True:
    main()
