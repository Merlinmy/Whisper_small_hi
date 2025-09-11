import os
import subprocess

# Folder where your HEVC files are stored
input_folder = "vishnu_videos"
output_folder = "output_mp4"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith((".mp4", ".mov", ".mkv", ".hevc")):
        input_path = os.path.join(input_folder, file_name)

        # Output file name (same name but .mp4)
        base_name, _ = os.path.splitext(file_name)
        output_path = os.path.join(output_folder, f"{base_name}_h264.mp4")

        # ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "128k",
            output_path
        ]

        print(f"Converting {input_path} → {output_path}")
        subprocess.run(cmd, check=True)

print("✅ All files converted successfully!")
