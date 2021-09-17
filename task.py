# Function
import datetime
import time
import webbrowser
from re import search

import pyautogui
import pywhatkit
import pywikihow
import wikipedia
from pyfirmata import OUTPUT, Arduino, util
from pyfirmata.pyfirmata import INPUT, Board

from listen import listen
from speak import say


board = Arduino('COM4')
board.digital[7].mode = INPUT
board.digital[4].mode = OUTPUT
board.digital[8].mode = OUTPUT
board.digital[12].mode = OUTPUT
# board.digital[7].write(1)
board.digital[4].write(1)
board.digital[8].write(1)
board.digital[12].write(1)


def wishme():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        say("good morning sir!")
    elif hour >= 12 and hour < 18:
        say("good afternoon sir!")
    else:
        say("good evening sir!")
    say("how may i help you sir. ")


def time():
    time = datetime.datetime.now().strftime("%H:%M")
    say(time)


def date():
    date = datetime.date.today()
    say(date)


def day():
    day = datetime.datetime.now().strftime("%A")
    say(day)


def screenshot():
    try:
        say("sir, please tell me the name for this screenshot file")
        name = listen().lower()
        say("please sir hold the screen for few seconds, I am taking screenshot")
        time.sleep(3)
        img = pyautogui.screenshot()
        img.save(f"{name}.png")
        say("ok done sir, The screenshot is saved in our main folder.")
    except Exception as e:
        say("soo sorry arjun sir. something was worng.")


def noninputfunction(query):
    query = str(query)

    if "time" in query:
        time()

    elif "date" in query:
        date()

    elif "day" in query:
        day()



def inputfuction(tag, query):

    if "wikipedia" in tag:
        try:
            name = str(query).replace("tell me", "")
            name = str(query).replace("skynet", "")
            name = str(query).replace("batao", "")
            name = str(query).replace("about", "")
            name = str(query).replace("tell something about", "")
            name = str(query).replace("what is", "")
            name = str(query).replace("who is", "")
            result = wikipedia.summary(name)
            say(result)
        except Exception as e:
            say("sorry, I don't know.")

    elif "google" in tag:
        query = str(query).replace("google", "")
        query = query.replace("search", "")
        pywhatkit.search(query)

    elif "screenshot" in query:
        screenshot()
    
    # elif "relay 2" in qu

