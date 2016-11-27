import cv2
from threading import Thread, Lock
from Queue import Queue
import sys
import numpy as np
import time, math
import tempfile
import pickle
import logging
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from os import path

# Parameter of the program
# _skip_rate = 4                # Number of frame to skip per frame
# _min_array_length = 150       # Depth when parallelising the program
# _min_scene_length = 6         # The minimum number of seconds in a scene
# _max_scene_length = 30        # The max number of seconds in a scene
# _alpha = 0.05                 # The weight for expaverage the scene detection

class Highlighter:
    @classmethod
    def load(cls, picklefile, datafile):
        return cls(load_model(picklefile, datafile))


    def __init__(self, model, _skip_rate=4, _max_array_ratio=0.27, _min_scene_length = 6, _max_scene_length=30,  _alpha = 0.05):
        self.params = object()
        self.skip_rate = _skip_rate
        self.max_array_ratio = _max_array_ratio
        self.min_scene_length = _min_scene_length
        self.max_scene_length = _max_scene_length
        self.alpha = _alpha
        
        self.model = model

    def get_highlights(self, filename, info):
        begin_time = time.time()

        cap = cv2.VideoCapture(filename)
        logging.info('loaded %i bytes from %s' % (path.getsize(filename), filename))

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if length < 5:
            raise ValueError("Video must be longer than 5 seconds, but is " + str(length))
        vid = {
            'length': length,
            'min_array_length': int(length * self.max_array_ratio),
            'fps': get_fps(cap)
        }
        lock = Lock()
        q = Queue()
        self.highlight_part(lock, cap, vid, 0, length, q)

        hl, ll, lr = q.get()
        
		# Clean the last and first scenes
        if len(hl) > 0:
            if hl[0]["duration"] <= self.min_scene_length or hl[0]["duration"] >= self.max_scene_length:
                del hl[0]
        if len(hl) > 0:
            if hl[-1]["duration"] <= self.min_scene_length or hl[-1]["duration"] >= self.max_scene_length:
                del hl[-1]

        i = 0
        avrg_duration = 0
        for scene in hl:
            vect = [
                float(info['category']),
                scene["avrg_brightness"],
                scene["avrg_bspeed"],
                scene["duration"],
                scene["max_brightness"],
                scene["max_bspeed"]
            ]
            vect = np.array(vect)
            avrg_duration += scene["duration"]
            #hl[i]["score"] = np.asscalar(self.model.predict(vect.reshape(1,-1)))
            i += 1

        if len(hl) > 0:
        	avrg_duration /= 1.0*len(hl)
        if avrg_duration > (self.max_scene_length + self.min_scene_length)/2:
        	hl.sort(key=lambda x: (-x['duration']))
        else:
        	hl.sort(key=lambda x: (x['start']))

        d = 0
        i = 0
        while i < len(hl):
            if d > self.max_scene_length:
            	del hl[i]
            else:
            	d += hl[i]['duration']
            i += 1

        return hl, (time.time() - begin_time)


# highlight_part(lock, cap, vid, 0, length, Queue())
		
    # recursively partition, analyize, and recombine information from a video file 		
    def highlight_part(self, lock, cap, vid, start, end, queue):
    	# Check If we reached bottom of recursion
    	logging.warn("vid['min_array_length'] " + str(vid['min_array_length']))
    	logging.warn("end-start " + str(end-start))
    	if end-start < vid['min_array_length']:
    		queue.put(self.conquer(lock, cap, vid, start, end))
    		return
        
    
    	# Divide part
    	queuer = Queue()
    	queuel = Queue()
    	q = (start+end)/2
    	self.highlight_part(lock, cap, vid, start, q, queuel)
    	self.highlight_part(lock, cap, vid, q, end, queuer)

    # 	threadl = Thread(target=self.highlight_part, args=(lock, cap, vid, start, q, queuel))
    # 	threadr = Thread(target=self.highlight_part, args=(lock, cap, vid, q, end, queuer))
    # 	threadl.start()
    # 	threadr.start()
    
    	# Waiting for threads to join
    # 	threadl.join()
    # 	threadr.join()
       
    	# Getting back the results
    	hl, lff, llf = queuel.get()
    	hr, rff, rlf = queuer.get()
    
    	# Combine part
    	if combine(llf, rff, hl[-1]["avrg_bspeed"]):
    		hl[-1]["end_frame"] = hl[0]["end_frame"]
    		hl[-1]["end"] = hl[0]["end"]
    		hl[-1]["duration"] += hl[0]["duration"]
    		hl[-1]["max_bspeed"] = max(hl[-1]["max_bspeed"], hl[0]["max_bspeed"])
    		hl[-1]["max_brightness"] = max(hl[-1]["max_brightness"], hl[0]["max_brightness"])
    		hl[-1]["avrg_brightness"] = (hl[-1]["avrg_brightness"] + hl[0]["avrg_brightness"])/2.
    		hl[-1]["avrg_bspeed"] = (hl[-1]["avrg_bspeed"] + hl[0]["avrg_bspeed"])/2.
    		del hr[0]
    
    	# Filtering the good scenes
    	ret_scenes = []
    	for j in range(len(hl)):
    		if j == 0 or (hl[j]["duration"] > self.min_scene_length and hl[j]["duration"] < self.max_scene_length):
    			ret_scenes.append(hl[j])
    	for j in range(len(hr)):
    		if  (j == len(hr) - 1) or (hr[j]["duration"] > self.min_scene_length and hr[j]["duration"] < self.max_scene_length):
    			ret_scenes.append(hr[j])
    
    	# Put the result in the return queue
        
    	queue.put((ret_scenes, lff, rlf))
    	return queue
	
	
    def conquer(self, lock, cap, vid, start, end):
    	# Background Subtraction parameters
    	i = 0
    	fgbg = cv2.createBackgroundSubtractorMOG2()
    
    	# Resutls
    	scenes = []
    
    	# Scene feature initialization
    	exp_avrg = 0
    	max_bspeed = 0
    	max_brightness = 0
    	avrg_brightness = 0
    
    	# Iterating over the frames
    	last_frame = []
    	first_frame = []
    	current_frame_start = start
    	currentframe = start
    	while currentframe < end:
    		# Locking the capture to get the frame
    		lock.acquire()
    		cap.set(cv2.CAP_PROP_POS_FRAMES, currentframe-1)
    		ret, frame = cap.read()
    		lock.release()
    
    		if ret:
    			# Save the first frame
    			if len(first_frame) == 0:
    				first_frame = frame
    
    			# Apply the background subtraction
    			if i == 2:
    				fgbg = cv2.createBackgroundSubtractorMOG2()
    				i = 0
    			fgmask = fgbg.apply(frame)
    			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 			width, height = frame.shape
    
    			# Get the brightness
    			brightness = frame.mean()
    			if i > 0 and i < 2:
    				w,h = fgmask.shape
    				diff = np.count_nonzero(fgmask)/(1.0*w*h)
    
    				# Add new scene if big difference encountered
    				if diff > exp_avrg or (currentframe + self.skip_rate+1) >= end:
    					if current_frame_start != currentframe:
    						avrg_brightness /= 1.0*(currentframe-current_frame_start)/(self.skip_rate+1)
    					scenes.append({"end_frame": currentframe, 
    								   "max_bspeed": max_bspeed,
    								   "avrg_bspeed": exp_avrg,
    								   "max_brightness": max_brightness,
    								   "avrg_brightness": avrg_brightness})
    
    					# Reinitialize the scene fetures
    					exp_avrg =  self.alpha*diff+(1- self.alpha)*exp_avrg
    					max_bspeed = 0
    					max_brightness = 0
    					avrg_brightness = 0
    					last_frame = frame
    
    				# Compute the frame features
    				max_bspeed = max(diff, max_bspeed)
    			
    			# Compute the frame features
    			max_brightness = max(brightness, max_brightness)
    			avrg_brightness += brightness
    			i += 1
    		
    		# Increment the frame index & skipping some
    		currentframe += self.skip_rate + 1
    	
    	# Save the scenes
    	fr_start = 0
    	ret_scenes = []
    	fps = vid['fps']
    	for j in range(len(scenes)):
    		scene = scenes[j]
    		fr_end = scene["end_frame"]
    		duration = (fr_end-fr_start)
    		if duration > self.skip_rate:
    			scenes[j]["start_frame"] = fr_start
    			scenes[j]["start"] = fr_start / fps
    			scenes[j]["end"] = fr_end / fps
    			scenes[j]["duration"] = duration / fps
    			scenes[j]["position"] = fr_start / vid['length']
    			ret_scenes.append(scenes[j])
    		fr_start = fr_end
    
    	# Return the result in the queue
    	return (ret_scenes, first_frame, last_frame)
	
def combine(frame1, frame2, exp_avrg):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg.apply(frame1)
    fgmask = fgbg.apply(frame2)
    w,h = fgmask.shape
    diff = np.count_nonzero(fgmask)/(1.0*w*h)
    return (diff > exp_avrg)
    
def get_fps(cap):
    fps = cap.get(cv2.CAP_PROP_POS_FRAMES)
    return 30 if(math.isnan(fps) or fps == 0) else fps

def load_model(pickename, dataname):
    try :
        import boto3
        bucket = boto3.client('s3').Bucket('highlightmodel')
        with tempfile.TemporaryFile(mode='w+b') as f:
            bucket.Object(pickename).download_fileobj(f)
            model = pickle.load(f)[0]
            logging.info("Loaded model from S3 pickle")
            return model
    except:
    	pass
    
    model_file = path.join(tempfile.gettempdir(),pickename)
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)[0]
            logging.info("Loaded model from S3 pickle")
            return model
    except:
        pass
    
    data = np.loadtxt(dataname, delimiter=',')
    X = data[:, 0:6]
    y = data[:, 6]
    model = MLPClassifier(
        activation='logistic', 
        max_iter=1000, 
        learning_rate='adaptive', 
        hidden_layer_sizes=np.full((7, ), 30))
    model.fit(X, y)
    with open(model_file, 'wb') as f:
        pickle.dump([model], f)
    return model

