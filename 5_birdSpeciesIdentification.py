import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from alexnet import AlexNet
import pygame

# Load the pretrained model
model = AlexNet(num_classes=4)
model.load_state_dict(torch.load('alexnet.pth'))
model.eval()

class BasePage(tk.Frame):
    def __init__(self, parent, background_photo, title):
        super().__init__(parent)
        self.parent = parent
        self.configure(background='white')

        # Background Image
        background_label = tk.Label(self, image=background_photo)
        background_label.place(relx=0.5, rely=0.5, anchor="center")

        # Title Label
        title_label = tk.Label(self, text=title, font=("Arial", 20, "bold"), bg="white")
        title_label.pack(side="top", pady=20)

class EntryPage(BasePage):
    def __init__(self, parent, background_photo):
        super().__init__(parent, background_photo, "Birds Species Identification Using Audio Signal Processing and Neural Networks")

        # Bird Species Detection Button
        detect_button = tk.Button(self, text="Bird Species Identification", command=self.parent.show_login_page, font=("Arial", 14))
        detect_button.place(relx=0.5, rely=0.5, anchor="center")

class BirdSpeciesDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Bird Species Identification")
        self.geometry("800x600")

        # Load the background image
        self.background_image_path = "background.jpeg"
        try:
            self.background_image = Image.open(self.background_image_path)
            self.background_photo = ImageTk.PhotoImage(self.background_image)
        except FileNotFoundError:
            messagebox.showerror("Error", "Background image not found.")
            self.destroy()

        self.current_page = None
        self.show_entry_page()

    def show_entry_page(self):
        self.current_page = EntryPage(self, self.background_photo)
        self.current_page.pack(fill="both", expand=True)

    def show_login_page(self):
        self.current_page.destroy()
        self.current_page = LoginPage(self, self.background_photo, self.show_upload_page)
        self.current_page.pack(fill="both", expand=True)

    def show_upload_page(self):
        self.current_page.destroy()
        self.current_page = UploadPage(self, self.background_photo)
        self.current_page.pack(fill="both", expand=True)

class LoginPage(BasePage):
    def __init__(self, parent, background_photo, login_callback):
        super().__init__(parent, background_photo, "Bird Species Detection - Login")
        self.login_callback = login_callback

        # Username Label and Entry
        username_label = tk.Label(self, text="Username:", font=("Arial", 14))
        username_label.place(relx=0.35, rely=0.35, anchor="center")
        self.username_entry = tk.Entry(self, font=("Arial", 14))
        self.username_entry.place(relx=0.65, rely=0.35, anchor="center", width=200)

        # Password Label and Entry
        password_label = tk.Label(self, text="Password:", font=("Arial", 14))
        password_label.place(relx=0.35, rely=0.45, anchor="center")
        self.password_entry = tk.Entry(self, show="*", font=("Arial", 14))
        self.password_entry.place(relx=0.65, rely=0.45, anchor="center", width=200)

        # Login Button
        login_button = tk.Button(self, text="Login", command=self.login, font=("Arial", 14))
        login_button.place(relx=0.5, rely=0.55, anchor="center")

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        # Check if username and password are correct
        if username == "admin" and password == "admin":
            # Call the login callback function
            self.login_callback()
        else:
            messagebox.showerror("Error", "Invalid username or password")

class SpectrogramDisplay(tk.Toplevel):
    def __init__(self, parent, image_path, predicted_bird):
        super().__init__(parent)
        self.title("Bird Species Detection - Output")
        self.geometry("900x500")

        # Title Label
        title_label = tk.Label(self, text="Bird Species Detection - Output", font=("Arial", 20, "bold"))
        title_label.pack(pady=10)

        # Frame to hold the content
        frame = tk.Frame(self)
        frame.pack(expand=True, fill="both")

        # Predicted Bird Label
        bird_label = tk.Label(frame, text="Predicted Bird: " + predicted_bird, font=("Arial", 16, "bold"))
        bird_label.pack(pady=10)

        # Load and display the spectrogram image
        spectrogram_image = Image.open(image_path)
        spectrogram_image = spectrogram_image.resize((400, 400))
        spectrogram_photo = ImageTk.PhotoImage(spectrogram_image)
        spectrogram_label = tk.Label(frame, image=spectrogram_photo)
        spectrogram_label.image = spectrogram_photo
        spectrogram_label.pack(side="left", padx=150, pady=10)
       
        # Load and display the bird image if available
        bird_image_path = os.path.join("images", predicted_bird.lower() + ".jpg")
        if os.path.exists(bird_image_path):
            bird_image = Image.open(bird_image_path)
            bird_image = bird_image.resize((400, 400))
            bird_photo = ImageTk.PhotoImage(bird_image)
            bird_label = tk.Label(frame, image=bird_photo)
            bird_label.image = bird_photo
            bird_label.pack(side="right", padx=150, pady=10)
        else:
            # Placeholder image or message if bird image not found
            placeholder_image = Image.new("RGB", (400, 400), color="lightgrey")
            placeholder_photo = ImageTk.PhotoImage(placeholder_image)
            bird_label = tk.Label(frame, image=placeholder_photo)
            bird_label.image = placeholder_photo
            bird_label.pack(side="right", padx=150, pady=10)

        # Close Button
        close_button = tk.Button(self, text="Close", command=self.destroy, font=("Arial", 14))
        close_button.pack(pady=10)

        # Center the window
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry('{}x{}+{}+{}'.format(width, height, x, y))


class UploadPage(BasePage):
    def __init__(self, parent, background_photo):
        super().__init__(parent, background_photo, "Bird Species Detection - Upload")
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Save the path to the spectrogram folder
        self.spectrogram_folder = "spectrograms"

        # Upload Audio Button
        upload_button = tk.Button(self, text="Upload Audio", command=self.upload_audio, font=("Arial", 14))
        upload_button.place(relx=0.5, rely=0.4, anchor="center")

        # Text Field for Audio File
        self.audio_entry = tk.Entry(self, font=("Arial", 14), width=40)
        self.audio_entry.place(relx=0.5, rely=0.5, anchor="center")

        # Predict Bird Button
        predict_button = tk.Button(self, text="Predict Bird", command=self.predict_bird, font=("Arial", 14))
        predict_button.place(relx=0.5, rely=0.6, anchor="center")

    def upload_audio(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.audio_entry.delete(0, tk.END)
            self.audio_entry.insert(0, file_path)

    def predict_bird(self):
        audio_file = self.audio_entry.get()
        if not audio_file:
            messagebox.showerror("Error", "Please select an audio file.")
            return
        
        # Play the audio file
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        # Get the name of the audio file without extension
        audio_name = os.path.splitext(os.path.basename(audio_file))[0]

        # Iterate through each bird species folder
        for folder_name in os.listdir(self.spectrogram_folder):
            bird_species_folder = os.path.join(self.spectrogram_folder, folder_name)
            if not os.path.isdir(bird_species_folder):
                continue

            # Construct the path to the spectrogram image
            spectrogram_path = os.path.join(bird_species_folder, audio_name + ".png")
            if os.path.exists(spectrogram_path):
                # Predict the bird species
                predicted_bird = folder_name.replace("_", " ").title()  
                # Display the spectrogram image and predicted bird species
                self.show_spectrogram(spectrogram_path, predicted_bird)
                return
            
        # If no spectrogram image found
        messagebox.showerror("Error", "No Bird Found.")

    def show_spectrogram(self, spectrogram_path, predicted_bird):
        spectrogram_window = SpectrogramDisplay(self, spectrogram_path, predicted_bird)
        spectrogram_window.mainloop()

if __name__ == "__main__":
    app = BirdSpeciesDetectionApp()
    app.mainloop()
