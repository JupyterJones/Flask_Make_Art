#!/home/jack/Desktop/Flask_Make_Art/flask_env/bin/python
import pygame
import glob

# Initialize the mixer module
pygame.mixer.init()

# Get a list of all MP3 files in the 'music' directory
music_files = glob.glob("/home/jack/Desktop/collections/MUSIC/*.mp3")

# Function to play a list of MP3 files sequentially
def play_music(files):
    for file in files:
        print(f"Now playing: {file}")
        pygame.mixer.music.load(file)  # Load the MP3 file
        pygame.mixer.music.play()  # Start playing the file

        # Wait until the current music finishes before moving to the next file
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

# Play the list of MP3 files
play_music(music_files)

# Optional: Clean up the mixer
pygame.mixer.quit()


