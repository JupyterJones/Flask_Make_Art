import sys
import os
import random
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip, ImageClip, vfx
from moviepy.config import change_settings
import shutil
change_settings({"FFMPEG_BINARY": "/usr/local/bin/ffmpeg"})  # Update path if needed

def run_ffmpeg(command):
    """Helper to run FFmpeg commands safely"""
    result = os.system(command)
    if result != 0:
        print(f"FFmpeg failed: {command}")
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python join2addmusicR.py <input_video1.mp4> <input_video2.mp4>")
        sys.exit(1)

    input_video1_path = sys.argv[1]
    input_video2_path = sys.argv[2]

    # Ensure the files exist
    if not os.path.exists(input_video1_path) or not os.path.exists(input_video2_path):
        print("Error: One or both video files do not exist.")
        sys.exit(1)

    try:
        # Step 1: Reverse the first video using FFmpeg
        print("Reversing the first video using FFmpeg...")
        reversed_video_path = "REV.mp4"
        run_ffmpeg(f"ffmpeg -i {input_video1_path} -vf reverse -af areverse -y {reversed_video_path}")

        # Step 2: Load videos into MoviePy
        print("Loading videos into MoviePy...")
        video1 = VideoFileClip(input_video1_path)
        reversed_video = VideoFileClip(reversed_video_path)
        video2 = VideoFileClip(input_video2_path)

        # Step 3: Create SEG1 (video1 + reversed video1)
        print("Creating SEG1...")
        seg1 = concatenate_videoclips([video1, reversed_video])
        seg1.write_videofile("SEG1.mp4", codec="libx264", audio_codec="aac")

        # Step 4: Create PHASE1 (SEG1 + video2)
        print("Creating PHASE1...")
        phase1 = concatenate_videoclips([seg1, video2])
        phase1.write_videofile("PHASE1.mp4", codec="libx264", audio_codec="aac")

        # Step 5: Add a random MP3 audio and a PNG frame
        print("Adding random audio and PNG overlay...")
        mp3_files = [f for f in os.listdir('.') if f.endswith('.mp3')]
        if not mp3_files:
            print("Error: No MP3 files found.")
            sys.exit(1)
        random_mp3 = random.choice(mp3_files)
        audio = AudioFileClip(random_mp3)

        png_path = "512x768.png"
        if not os.path.exists(png_path):
            print(f"Error: PNG file {png_path} not found.")
            sys.exit(1)

        # Resize video to fit within the frame (padding effect)
        print("Adding frame to the video...")
        frame_clip = ImageClip(png_path).set_duration(phase1.duration)
        video_resized = phase1.resize(height=720, width=510)  # Adjust size to fit within the frame
        result = CompositeVideoClip([frame_clip, video_resized.set_position((25,20))]).set_audio(audio)

        result.write_videofile("RESULT.mp4", codec="libx264", audio_codec="aac")

        # Step 6: Slow down the video
        print("Slowing down the video...")
        slowed_result = result.fx(vfx.speedx, 0.5)
        slowed_result.write_videofile("SLOWED_RESULT.mp4", codec="libx264", audio_codec="aac")
        shutil.copy("SLOWED_RESULT.mp4", "../video_history/RESULT.mp4")
        print("✅ All steps completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
