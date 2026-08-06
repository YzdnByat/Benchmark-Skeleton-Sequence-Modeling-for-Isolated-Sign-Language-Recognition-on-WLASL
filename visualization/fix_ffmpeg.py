import imageio_ffmpeg
import shutil
import os

# Find where the hidden ffmpeg executable is stored
hidden_ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

# Define the destination path (right next to your app.py)
project_dir = os.path.dirname(os.path.abspath(__file__))
destination = os.path.join(project_dir, "ffmpeg.exe")

# Copy and rename it
shutil.copy(hidden_ffmpeg_path, destination)

print(f"✅ SUCCESS! Copied FFmpeg to: {destination}")
print("Gradio will now find it automatically. You can run app.py again.")