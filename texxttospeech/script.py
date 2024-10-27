from gtts import gTTS
import pygame
import tkinter as tk
from tkinter import messagebox

# Function to convert text to speech and play it using pygame
def speak_text():
    text = entry.get()  # Get the text from the entry box
    if text.strip() == "":  # Check if the input is empty
        messagebox.showwarning("Input Error", "Please enter some text to speak.")
        return

    # Convert text to speech and save it as an audio file
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")

    # Initialize pygame mixer and play the audio
    pygame.mixer.init()
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()

    # Keep the program running until the audio finishes
    while pygame.mixer.music.get_busy():
        pass

# Create the main window
root = tk.Tk()
root.title("Text-to-Speech")

# Create and place the text entry widget
entry = tk.Entry(root, width=50, font=("Arial", 14))
entry.pack(pady=20)

# Create and place the Speak button
speak_button = tk.Button(root, text="Speak", command=speak_text, font=("Arial", 14), bg="blue", fg="white")
speak_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
