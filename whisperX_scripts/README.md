Forced Alignment Pipeline
=========================

Generates per-word timestamps from video + known script using WhisperX
forced alignment (wav2vec2), then produces ASS subtitles with karaoke
tags and burns them onto the video.


Project layout
--------------
batch20/                  Input videos ({ITM_ID}_*_script{1|2}_*.mp4)
300k/                     Script JSONL files (scripts_batch_NNNN.jsonl)
alignments/batch20/       Output: per-video .json and .ass files
outputs/batch20/          Output: burned subtitle videos
align.py                  Core alignment module (audio extraction + WhisperX)
json_to_ass.py            Converts word-timestamp JSON to ASS karaoke format
burn_subs.py              Burns ASS subtitles onto video via ffmpeg
run_batch20.py            Batch runner for the full pipeline
requirements.txt          Python dependencies


Prerequisites
-------------
- Python 3.11
- ffmpeg (brew install ffmpeg)


Setup
-----
python3.11 -m venv venv
venv/bin/pip install -r requirements.txt


Run (full batch)
----------------
venv/bin/python3 run_batch20.py

Processes every .mp4 in batch20/, skips videos already done.
Per-video outputs:
  alignments/batch20/{stem}.json   word-level timestamps + scores
  alignments/batch20/{stem}.ass    ASS subtitle file with {\k} karaoke tags
  outputs/batch20/{stem}.mp4       video with subtitles burned in


Script lookup
-------------
Each video filename encodes the item ID and script number:
  ITM0033372FB3E91_male_selfie_9_male-5_script2_Bottle.mp4
                                          ^^^^^^^
The runner looks up item_id + script{1|2} in scripts_batch_0020.jsonl
to get the transcript used for forced alignment.


ASS subtitle format
-------------------
Matches the reference style (Arial 48, green, PlayResX=384):
  Dialogue: 0,0:00:0.50,0:00:1.44,Default,,0,0,0,,{\k8}If {\k18}your {\k42}garden {\k26}feels

{\kN} = word highlight duration in centiseconds.
Lines are grouped at natural pauses (>=0.3s) or every 4 words.


Using modules individually
--------------------------
# Align one audio file against a known script
from align import extract_audio, align_audio
extract_audio("video.mp4", "audio.wav")
words = align_audio("audio.wav", "The script text goes here.")

# Convert alignment JSON to ASS
from json_to_ass import json_to_ass
json_to_ass("alignment.json", "subtitles.ass")

# Burn subtitles onto video
from burn_subs import burn_subtitles
burn_subtitles("video.mp4", "subtitles.ass", "output.mp4")


Performance (CPU, Apple Silicon)
---------------------------------
~5.4s wall time per ~13s video (wav2vec2 model loads once per process).
Full 38-video batch: ~94s total.
