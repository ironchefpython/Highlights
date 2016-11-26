import youtube_dl
from pytube import YouTube
import numpy as np
import cv2 as cv2
import urllib
import json, uuid
import pickle
from threading import Thread
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import urlparse
from highlighter3 import Highlighter
import tempfile
import logging
from os import path, environ

logging.basicConfig(level=logging.DEBUG)

_api_key = environ['API_KEY']
YT_OPTS = {
    'format': '17',       
    'outtmpl': path.join(tempfile.mkdtemp(), 'vid_%(id)s.3gp'),        
    'noplaylist' : True
}

HIGHLIGHTER = Highlighter.load('nn_model.pickle', 'data_out.csv')

# Given a video id, downlaods the video and returns the filename
def get_video(video_id):
    url = "https://www.youtube.com/watch?v=%s" % (video_id,)
    tempfile = YT_OPTS['outtmpl'] % {'id': video_id}
    with youtube_dl.YoutubeDL(YT_OPTS) as ydl:
	    ydl.download([url])
    logging.debug("downloaded video from %s to %s" % (url, tempfile))
    return tempfile
	
# Extract the highlights from one video
def get_highlights(filename, vid):
    hl = HIGHLIGHTER.get_highlights(filename, vid)
    logging.debug("generated highlights from %s" % (filename,))
    return hl

def get_item(id):
    url = 'https://www.googleapis.com/youtube/v3/videos?part=snippet&id='+id+'&key='+_api_key
    data = json.loads(urllib.urlopen(url).read())
    logging.debug("fetched video data from %s" % (url,))
    return data["items"][0]


class Spot:
    def __init__(self, video_id):
        snippet = get_item(video_id)['snippet']
        self.vid = {
            'videoId': video_id,
            'snippet': snippet,
            'category': snippet['categoryId']
        }
        self.vid.update(get_highlights(get_video(video_id), self.vid))
                        
    def toJson(self):
        return json.dumps(self.vid)

