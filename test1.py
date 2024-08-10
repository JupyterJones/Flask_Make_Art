import yt_dlp
import os

def download_youtube_video(url, download_folder):
    try:
        ydl_opts = {
            'outtmpl': os.path.join(download_folder, '%(title)s.%(ext)s'),
            'format': 'bestvideo+bestaudio/best',
            'noplaylist': True  # Download only the single video, not the playlist
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            print(f"Downloading video: {info_dict.get('title')}")
            print(f"Video downloaded to {download_folder}")
    except Exception as e:
        print(f"Error downloading video: {e}")

if __name__ == '__main__':
    DOWNLOAD_FOLDER = 'downloads'
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    video_url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'  # Replace with your video URL
    download_youtube_video(video_url, DOWNLOAD_FOLDER)
