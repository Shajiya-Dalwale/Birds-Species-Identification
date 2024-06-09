import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def create_spectrogram(audio_file_path, save_path):
    # Load the audio file
    y, sr = librosa.load(audio_file_path)
    
    # Compute the spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    
    # Display the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    # Save the spectrogram image
    plt.savefig(save_path)
    
    # Show the plot
    plt.show()

# Provide the paths for the audio file and save location
audio_file_path = input("Enter the path of the audio file: ")
#C:\BSD\dataset\cuckoo\XC33104 - Indian Cuckoo - Cuculus micropterus concretus.mp3
save_path = input("Enter the path to save the spectrogram image: ")
#C:\BSD\dataset\cuckoo\XC33104 - Indian Cuckoo - Cuculus micropterus concretus.png
# Call the function to create the spectrogram
create_spectrogram(audio_file_path, save_path)
print("Spectrogram Created and Saved Succesfully!")
