#test.py
from pytube import YouTube
import os

def download_youtube_video(url, download_folder):
    try:
        yt = YouTube(url)
        # Get the highest resolution stream available
        stream = yt.streams.get_highest_resolution()
        print(f"Downloading video from {url}")
        
        # Download the video
        video_path = stream.download(output_path=download_folder)
        
        # Print the path where the video was saved
        print(f"Video downloaded to {video_path}")
        return video_path
    except Exception as e:
        print(f"Error downloading video: {e}")

if __name__ == '__main__':
    # Define a folder to save the downloaded video
    DOWNLOAD_FOLDER = 'static/downloads'
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    
    # Replace with a valid YouTube video URL
    video_url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'  # Example URL (Rick Astley - Never Gonna Give You Up)
    video_url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    # Call the function to download the video
    download_youtube_video(video_url, DOWNLOAD_FOLDER)
