import os
from moviepy.editor import *

def extract_frames(movie, times, imgdir):
    clip = VideoFileClip(movie)
    for t in times:
        imgpath = os.path.join(imgdir, '{}.png'.format(t))
        clip.save_frame(imgpath, t)

movie = '../project_video.mp4'
imgdir = 'output_frames'
times = range(1,50)

extract_frames(movie, times, imgdir)