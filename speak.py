import pyttsx3
# from task import wishme

def say(Text):
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    engine.setProperty('voices', voices[1].id)
    engine.setProperty('rate', 165)
    print("    ")
    print(f"A.I : {Text}")
    engine.say(text = Text)
    engine.runAndWait()
    print("    ")


