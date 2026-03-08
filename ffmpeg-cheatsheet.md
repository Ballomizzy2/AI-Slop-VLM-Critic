# FFmpeg Cheat Sheet
> Multimodal Hackathon Edition

---

## 🎞 Frame Extraction

```bash
# Extract 1 frame per second
ffmpeg -i input.mp4 -vf fps=1 frames/%04d.jpg

# Extract every N frames (e.g. every 30)
ffmpeg -i input.mp4 -vf "select=not(mod(n\,30))" -vsync vfr frames/%04d.jpg

# Extract a single frame at timestamp
ffmpeg -ss 00:00:05 -i input.mp4 -frames:v 1 frame.jpg

# Extract frames as high-quality PNG
ffmpeg -i input.mp4 -vf fps=1 -q:v 1 frames/%04d.png
```

---

## 🎙 Audio Extraction

```bash
# Extract audio as MP3
ffmpeg -i input.mp4 -q:a 0 -map a output.mp3

# Extract audio as WAV (for Whisper)
ffmpeg -i input.mp4 -ar 16000 -ac 1 output.wav

# Strip audio (video only)
ffmpeg -i input.mp4 -an no_audio.mp4
```

---

## ✂️ Trimming & Clipping

```bash
# Trim by start time + duration
ffmpeg -ss 00:00:10 -i input.mp4 -t 30 -c copy clip.mp4

# Trim by start + end timestamp
ffmpeg -ss 00:00:10 -to 00:00:40 -i input.mp4 -c copy clip.mp4

# Re-encode clip (slower but more accurate)
ffmpeg -i input.mp4 -ss 00:00:10 -t 30 clip.mp4
```

---

## 🔍 Video Inspection

```bash
# Get full video metadata
ffprobe -v quiet -print_format json -show_format -show_streams input.mp4

# Get duration only
ffprobe -v error -show_entries format=duration -of csv=p=0 input.mp4

# Get resolution (width x height)
ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of json input.mp4

# Get FPS
ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate input.mp4
```

---

## 🔄 Conversion & Format

```bash
# Convert to MP4 (H.264)
ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4

# Convert video to GIF
ffmpeg -i input.mp4 -vf "fps=10,scale=640:-1" output.gif

# Images → video (slideshow)
ffmpeg -framerate 1 -i frames/%04d.jpg -c:v libx264 output.mp4

# Resize video
ffmpeg -i input.mp4 -vf scale=1280:720 output.mp4
```

---

## 🎬 Scene Detection

```bash
# Detect scene cuts and save frames
ffmpeg -i input.mp4 -vf "select='gt(scene,0.3)',showinfo" -vsync vfr scenes/%04d.jpg 2>&1 | grep showinfo

# Extract keyframes only
ffmpeg -i input.mp4 -vf "select=eq(pict_type\,I)" -vsync vfr keyframes/%04d.jpg
```

---

## 🔗 Concat & Merge

```bash
# Concatenate clips (same codec)
ffmpeg -f concat -safe 0 -i list.txt -c copy output.mp4

# list.txt format:
# file 'clip1.mp4'
# file 'clip2.mp4'
# file 'clip3.mp4'

# Merge video + audio track
ffmpeg -i video.mp4 -i audio.mp3 -c:v copy -map 0:v:0 -map 1:a:0 output.mp4
```

---

## ⚡ Useful Flags

```bash
# Overwrite output without asking
ffmpeg -y -i input.mp4 output.mp4

# Suppress verbose output
ffmpeg -v quiet -stats -i input.mp4 output.mp4

# Fast copy (no re-encode)
ffmpeg -i input.mp4 -c copy output.mp4

# Set video quality (lower = better, range 18–28)
ffmpeg -i input.mp4 -crf 23 output.mp4
```

---

> 💡 **Hackathon tip:** For VLM critic pipeline → extract keyframes with `fps=1`, send to Claude vision, score per frame, flag bad ones.
