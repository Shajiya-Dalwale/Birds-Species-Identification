import tkinter as tk
from tkinter import filedialog
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def remove_silence(audio, threshold=0.02, frame_length=1024):
    # Find non-silent intervals
    non_silent_intervals = librosa.effects.split(audio, top_db=threshold, frame_length=frame_length)
    
    # Concatenate non-silent intervals
    non_silent_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
    
    return non_silent_audio

def pad_audio(audio, target_length):
    # Pad or truncate audio to match target length
    if len(audio) < target_length:
        padded_audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    else:
        padded_audio = audio[:target_length]
    
    return padded_audio

def plot_vibration_graph(original_audio, processed_audio, noise_audio, sr):
    plt.figure(figsize=(10, 6))

    # Plot original audio
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(original_audio)) / sr, original_audio)
    plt.title('Original Audio Signal')

    # Plot processed audio (after silence removal)
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(processed_audio)) / sr, processed_audio)
    plt.title('Signal after Silence Removal')

    # Plot noise
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(noise_audio)) / sr, noise_audio)
    plt.title('Noise of Audio')

    plt.tight_layout()
    plt.suptitle('Silence Removal and Reconstruction of the Audio Signal', y=0.02)
    plt.show()

def upload_file():
    file_path = filedialog.askopenfilename()
    audio_text.delete(0, tk.END)
    audio_text.insert(0, file_path)

def process_audio():
    audio_path = audio_text.get()
    if audio_path:
        y, sr = librosa.load(audio_path, sr=None)
        y_length = len(y)
        processed_length = int(0.025 * y_length)  # 2.5% of audio length
        y_processed = remove_silence(y[:processed_length])
        y_processed = pad_audio(y_processed, processed_length)
        noise_audio = y[:processed_length] - y_processed
        plot_vibration_graph(y[:processed_length], y_processed, noise_audio, sr)

# Create a GUI window
root = tk.Tk()
root.title("Audio Signal Reconstruction")
root.configure(bg="#ffe4c4")  # Peach background

# Center the window on the screen
window_width = 400
window_height = 200
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = (screen_width / 2) - (window_width / 2)
y_coordinate = (screen_height / 2) - (window_height / 2)
root.geometry(f"{window_width}x{window_height}+{int(x_coordinate)}+{int(y_coordinate)}")

# Create and place widgets for the login page
frame = tk.Frame(root, bg="#ffe4c4")  # Peach background
frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Create a text field for displaying the selected file path
audio_text = tk.Entry(root, width=50)
audio_text.pack(pady=10)

# Create an upload button
upload_button = tk.Button(root, text="Upload a Audio for Scilence Removal and Reconstruction of the Audio Signal", command=upload_file)
upload_button.pack(pady=5)

# Create a submit button
submit_button = tk.Button(root, text="Submit", command=process_audio)
submit_button.pack(pady=5)

# Run the GUI
root.mainloop()
