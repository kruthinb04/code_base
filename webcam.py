import os
import cv2
import uuid
import gradio as gr
import numpy as np
import pyttsx3
import threading
import speech_recognition as sr
from threading import Lock

import webcamgpt

MARKDOWN = """
# XTRAIL GEN-AI Solutions
"""

connector = webcamgpt.OpanAIConnector()
engine = pyttsx3.init()
recognizer = sr.Recognizer()
engine_lock = Lock()

def save_image_to_drive(image: np.ndarray) -> str:
    if image is None or not np.any(image):
        print("No image data to save.")
        return None
    image_filename = f"{uuid.uuid4()}.jpeg"
    image_directory = "data"
    if not os.path.exists(image_directory):
        os.makedirs(image_directory, exist_ok=True)
    image_path = os.path.join(image_directory, image_filename)
    if image.dtype != np.uint8:
        image = (255 * image).astype(np.uint8)
    cv2.imwrite(image_path, image)
    if not os.path.exists(image_path):
        print("Failed to save the image.")
        return None
    return image_path

def speak(text):
    with engine_lock:  # Acquire the lock before using the speech engine
        engine.say(text)
        engine.runAndWait()

def stop_speaking():
    with engine_lock:
        engine.stop()  # This method stops the current and queued utterances

def respond(image: np.ndarray, prompt: str, chat_history):
    if image is None:
        print("Received an empty image.")
        return "", chat_history
    image = cv2.cvtColor(np.fliplr(image), cv2.COLOR_RGB2BGR)
    image_path = save_image_to_drive(image)
    if not image_path:
        response = "Failed to process image."
    else:
        response = connector.simple_prompt(image=image, prompt=prompt)
        chat_history.append(((image_path,), None))
        chat_history.append((prompt, response))
        threading.Thread(target=speak, args=(response,)).start()
    return "", chat_history

def listen_and_respond(image: np.ndarray, chat_history):
    while True:
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
            try:
                prompt = recognizer.recognize_google(audio)
                print(f"Recognized: {prompt}")
                _, chat_history = respond(image, prompt, chat_history)
            except sr.UnknownValueError:
                error_message = "Sorry, I did not understand that."
                threading.Thread(target=speak, args=(error_message,)).start()
            except sr.RequestError:
                error_message = "Sorry, the speech service is down."
                threading.Thread(target=speak, args=(error_message,)).start()

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        webcam = gr.Image(source="webcam", streaming=True)
        with gr.Column():
            chatbot = gr.Chatbot(height=500)
            message = gr.Textbox()
            speak_button = gr.Button("Start Listening")
            stop_button = gr.Button("Stop Speaking", on_click=stop_speaking)
            clear_button = gr.ClearButton([message, chatbot])

    message.submit(respond, [webcam, message, chatbot], [message, chatbot])
    speak_button.click(lambda: threading.Thread(target=listen_and_respond, args=(webcam, chatbot)).start(), [], [])

demo.launch(debug=True, show_error=True)
