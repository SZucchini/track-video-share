import os
import argparse
from logging import getLogger, StreamHandler, DEBUG, Formatter

import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms

logger = getLogger("Log")
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(handler_format)
logger.addHandler(handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolov8 = YOLO("./models/yolov8x-seg.pt")
yolov8.to(device)
logger.debug("YOLOv8 loaded")

vit = torch.load('./models/vit-classifier.pth')
vit.eval()
vit.to(device)
logger.debug("ViT loaded")

img_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def get_video(input):
    video = cv2.VideoCapture(input)
    fps = video.get(cv2.CAP_PROP_FPS)
    w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return video, fps, w, h, fourcc


def check_runner(frame):
    results = yolov8(frame, max_det=12, conf=0.5, classes=0, save=False, retina_masks=True)
    if len(results[0]) == 0:
        return False

    imgs = []
    torch_imgs = []
    for i in range(len(results[0])):
        bbox = results[0].boxes.xyxy[i].to("cpu").numpy().astype(int)
        mask = results[0].masks.data[i].to("cpu").numpy()
        masked = (frame * mask[:, :, None])[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        img_masked = Image.fromarray(masked.astype('uint8'))
        imgs.append(img_masked)
        torch_imgs.append(img_transforms(img_masked))

    torch_imgs = torch.stack(torch_imgs).to(device)
    with torch.no_grad():
        output = vit(torch_imgs).argmax(dim=1)
    np_output = output.to("cpu").numpy()

    if np.any(np_output == 1):
        return True
    else:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input video path")
    parser.add_argument("--output", type=str, required=True, help="output video dir")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    video, fps, w, h, fourcc = get_video(args.input)

    exist = False
    write = False
    finish = 0
    skip = 0
    cnt_frame = 0
    while True:
        ret, frame = video.read()
        if not ret:
            writer.release()
            break

        if finish == -1 and skip < 60:
            skip += 1
            cnt_frame += 1
            continue

        if cnt_frame % 4 == 0:
            exist = check_runner(frame)

        if exist and not write:
            write = True
            finish = 0
            output = args.output + f'/trimed_{cnt_frame}.mp4'
            writer = cv2.VideoWriter(output, fourcc, int(fps), (int(w), int(h)))

        if not exist and write:
            if finish < 90:
                finish += 1
            else:
                writer.release()
                write = False
                finish = -1
                skip = 0

        if write:
            writer.write(frame)

        logger.debug("frame {} processed".format(cnt_frame))
        cnt_frame += 1

    video.release()


if __name__ == '__main__':
    main()
