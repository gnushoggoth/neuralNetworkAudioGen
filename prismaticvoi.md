# Stitching MP4 Videos and Merging with Audio on macOS (FFmpeg)

## **Prerequisites**
Ensure **Homebrew** is installed (or prepare for the wrath of the cosmic void that is dependency hell). You can check if it's already installed with:

```sh
command -v brew
```

If not, install it with: If not, install it with:

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install **FFmpeg**:

```sh
brew install ffmpeg  # Installs FFmpeg with default options. If you need additional codecs, use `brew install ffmpeg --with-options`.
```

---

## **Step 1: Concatenating Multiple MP4 Files**

### **1.1 Create a File List**
If all `.mp4` files are encoded the same way (same codec, resolution, etc.), create a text file named `file_list.txt` with the following content:

```
file 'video1.mp4'
file 'video2.mp4'
file 'video3.mp4'
```

### **1.2 Run FFmpeg to Concatenate**

```sh
ffmpeg -f concat -safe 0 -i file_list.txt -c copy output.mp4 # Let the arcane rituals of concatenation commence.
```

⚠️ **If the MP4 files have different encodings, you need to re-encode:**

```sh
ffmpeg -i "video1.mp4" -i "video2.mp4" -filter_complex "[0:v:0][1:v:0]concat=n=2:v=1[outv]" -map "[outv]" -c:v libx264 output.mp4
```

---

## **Step 2: Merging MP4 Video with Audio (MP3/OGG)**

### **2.1 Merge Video with MP3 Audio**

```sh
ffmpeg -i output.mp4 -i audio.mp3 -c:v copy -c:a aac -strict experimental final_video.mp4
```

### **2.2 Merge Video with OGG Audio**

```sh
ffmpeg -i output.mp4 -i audio.ogg -c:v copy -c:a aac -strict experimental final_video.mp4
```

### **2.3 If Audio is Shorter or Longer than Video**

To automatically adjust length to the shorter of the two:

```sh
ffmpeg -i output.mp4 -i audio.mp3 -c:v copy -c:a aac -shortest final_video.mp4
```

---

## **Step 3: Automate with a Bash Script**

Save the following script as `stitch_videos.sh`:

```sh
#!/bin/bash

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg is not installed. Install it with: brew install ffmpeg"
    exit 1
fi

# Ensure input files exist
if [ ! -f "$1" ] || [ ! -f "$2" ]; then
    echo "Usage: $0 <video_file.mp4> <audio_file.mp3|ogg>"
    exit 1
fi

VIDEO=$1
AUDIO=$2
OUTPUT="final_video.mp4"

# Merge video and audio
ffmpeg -i "$VIDEO" -i "$AUDIO" -c:v copy -c:a aac -strict experimental "$OUTPUT"

echo "Done! Merged file saved as $OUTPUT"
```

### **Make It Executable & Run**

```sh
chmod +x stitch_videos.sh # Empower the script with the eldritch right to execute.
./stitch_videos.sh output.mp4 audio.mp3
```

---

## **Python Alternative (If Needed)**

If Python is installed, install `moviepy`:

```sh
# If Python is installed:
pip install moviepy # Summon the digital spirit of video manipulation.

# Alternatively, if you don't want to use Python, you can use FFmpeg instead:
ffmpeg -i output.mp4 -i audio.mp3 -c:v copy -c:a aac -strict experimental final_video.mp4
```

Save this script as `stitch_videos.py`:

```python
from moviepy.editor import VideoFileClip, AudioFileClip
import sys

if len(sys.argv) != 3:
    print("Usage: python stitch_videos.py <video.mp4> <audio.mp3|ogg>")
    sys.exit(1)

video = VideoFileClip(sys.argv[1])
audio = AudioFileClip(sys.argv[2])

final_video = video.set_audio(audio)
final_video.write_videofile("final_output.mp4", codec="libx264", audio_codec="aac")
```

Run it with:

```sh
python -c "import importlib.util; import sys; package='moviepy';
if importlib.util.find_spec(package) is None:
    print(f'Error: {package} is not installed. Install it with: pip install {package}');
    sys.exit(1)"
stitch_videos.py output.mp4 audio.mp3
```

---

## **Final Thoughts**
- **For maximum reliability and speed, use FFmpeg.**
- **If you prefer scripting in Python, `moviepy` is a simple alternative.**

🔮 **SREs don’t have time for mortal struggles—this is a forbidden incantation of automation.** 🚀💀

