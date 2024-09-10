import cv2
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import time

# Configs
ASCII_CHARS = ['@', '#', 'S', '%', '?', '*', '+', ';', ':', ',', '.']
ASCII_WIDTH = 100
FRAME_DELAY = 1 / 30



def resize_image(image, new_width=ASCII_WIDTH):
    width, height = image.size
    ratio = height / width / 1.65
    new_height = int(new_width * ratio)
    resized_image = image.resize((new_width, new_height))
    return resized_image

def grayify(image):
    grayscale_image = ImageOps.grayscale(image)
    return grayscale_image

def pixels_to_ascii(image):
    pixels = np.array(image)
    ascii_str = np.array(ASCII_CHARS)[pixels // 25]
    ascii_str = "\n".join(["".join(row) for row in ascii_str])
    return ascii_str

def image_to_ascii(image, width=ASCII_WIDTH):
    image = resize_image(image, width)
    image = grayify(image)
    ascii_str = pixels_to_ascii(image)
    return ascii_str

def display_ascii_on_frame(frame, ascii_str):
    h, w, _ = frame.shape
    blank_image = np.zeros((h, w // 2, 3), np.uint8)

    y0, dy = 10, 10
    for i, line in enumerate(ascii_str.splitlines()):
        y = y0 + i * dy
        cv2.putText(blank_image, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    return blank_image

cap = cv2.VideoCapture(0)
fig, ax = plt.subplots()
prev_frame_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    curr_frame_time = time.time()
    time_elapsed = curr_frame_time - prev_frame_time

    if time_elapsed < FRAME_DELAY:
        continue  
    prev_frame_time = curr_frame_time
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ascii_art = image_to_ascii(pil_image)
    ascii_frame = display_ascii_on_frame(frame, ascii_art)
    combined_frame = np.hstack((frame, ascii_frame))
    ax.clear()
    ax.imshow(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.001)
    
    if plt.get_fignums() == []:
        break

cap.release()
plt.close()
