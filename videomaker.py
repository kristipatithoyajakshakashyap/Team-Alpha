from moviepy.editor import *
from pathlib import Path

img_clips = []
path_list=[]

#accessing path of each image
for image in os.listdir('images/'):
    if image.endswith(".jpg"):
        path_list.append(os.path.join('images/', image))

#creating slide for each image
for img_path in path_list:
  slide = ImageClip(img_path,duration=2)
  img_clips.append(slide)

#concatenating slides
video_slides = concatenate_videoclips(img_clips, method='compose')
#exporting final video
video_slides.write_videofile("output_video.mp4", fps=24)