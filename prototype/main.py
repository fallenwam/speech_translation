import os
import time
import deepl
import speech_recognition as sr
from gtts import gTTS
import pygame
from dotenv import load_dotenv

load_dotenv()
AUTH_KEY = os.getenv("DEEPL_AUTH_KEY")

def translate_to_text():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

        print("\n--- 1. Listening ---\n")
        audio = recognizer.listen(source)

    print("Processing audio...\n")
    try:
        text = recognizer.recognize_google(audio)
        print(f"[Input: {text}]\n")
        return text
    
    except sr.UnknownValueError:
        return "[Error: Google Speech Recognition could not understand audio]"
    except sr.RequestError as e:
        return f"Error: Could not request results from Google Speech Recognition service; {e}"


def translate_to_language(text_to_translate, target_language):
    deepl_client = deepl.DeepLClient(AUTH_KEY)

    print("\n--- 2. Translating ---\n")

    result = deepl_client.translate_text(text_to_translate, target_lang=target_language)
    print(f"Translation: {result.text}\n")
    return result.text

def translate_to_speech(text,lang_code):
    print("\n--- 3. Speaking ---\n")

    tts = gTTS(text=text, lang=lang_code)
    filename = "speech_output.mp3"
    tts.save(filename)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    pygame.mixer.quit()
    time.sleep(0.1)

    if os.path.exists(filename):
        os.remove(filename)
    print("\nDone!\n")



    

if __name__ == "__main__":
    SUPPORTED_LANGUAGES = {
        "japanese": {"deepl": "JA", "gtts": "ja"},
        "french":   {"deepl": "FR", "gtts": "fr"},
        "spanish":  {"deepl": "ES", "gtts": "es"},
        "hebrew":   {"deepl": "HE", "gtts": "iw"},
        "german":   {"deepl": "DE", "gtts": "de"}
    }

    print("Supported languages:", ", ".join(SUPPORTED_LANGUAGES.keys()).title())
    user_choice = input("What language would you like to translate to? ").strip().lower()

    if user_choice not in SUPPORTED_LANGUAGES:
        print(f"Error: '{user_choice}' is not currently supported. Exiting.")
        exit()

    target_deepl = SUPPORTED_LANGUAGES[user_choice]["deepl"]
    target_gtts = SUPPORTED_LANGUAGES[user_choice]["gtts"]

    text = translate_to_text()
    if not text.startswith("[Error"):
        translated = translate_to_language(text, target_deepl)

        translate_to_speech(translated,target_gtts)