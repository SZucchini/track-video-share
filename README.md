# track-video-share

## Introduction

You can automatically detect passing scenes of runners from running training videos with this code.

## Author
Tomohiro Suzuki - suzuki.tomohiro@g.sp.m.is.nagoya-u.ac.jp

## How to use
### Step 0: Downloading the pre-trained model

Please download pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1HkCp8NINFvAqEfVfQmCoZ611ii9Idsou?usp=share_link) and put it in `./models`.

### Step 1: Prepare your input video

Prepare your original video and put it in `./input`.

### Step 2: Execute `./run.sh`

`sh ./run.sh -i <input video> -d <output directory>`.
