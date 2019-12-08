# ------------------------------
# Notice
# ------------------------------

# Copyright 2018 Clayton Darwin claytondarwin@gmail.com

# ------------------------------
# Imports
# ------------------------------
#Port connection info: bottom - right camera, top - left camera
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os,time,traceback
import threading,queue
import math
import numpy as np
import cv2
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

#cap = cv2.VideoCapture(0)


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v2_oid_v4_2018_12_12'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'oid_v4_label_map.pbtxt')

NUM_CLASSES = 601


# ## Download Model

#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
#print(category_index)


# ## Helper code

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
def get_id_for_label(val):
    for key,value in category_index.items(): # to find id of glasses
        for inner_key,inner_value in value.items():
          if inner_value==val:
            return value['id']


# # Detection

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# For synchronization of imshow
turn = 0

# ------------------------------
# Testing
# ------------------------------

def run():

    # ------------------------------
    # full error catch 
    # ------------------------------
    try:

        # ------------------------------
        # set up cameras 
        # ------------------------------

        # cameras variables
        left_camera_source = 2
        right_camera_source = 1
        pixel_width = 1280     # Based on specifications of C270 hd logitech camera  
        pixel_height = 720     
        angle_width = 46    #Found using manual methods (as seen in the video)       
        angle_height = 36
        frame_rate = 30                 #cameras tested with find_fps_webcam.py - keeps changing
        camera_separation = 2.8     #70 mm baseline separation (given in inches)
        
        # left camera 1
        ct1 = Camera_Thread()    # Each of these threads need to run both the SSD and the triangulation code
        ct1.camera_source = left_camera_source
        ct1.camera_width = pixel_width
        ct1.camera_height = pixel_height
        ct1.camera_frame_rate = frame_rate

        # right camera 2
        ct2 = Camera_Thread()
        ct2.camera_source = right_camera_source
        ct2.camera_width = pixel_width
        ct2.camera_height = pixel_height
        ct2.camera_frame_rate = frame_rate

        # camera coding
        ct1.camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        ct2.camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        # start cameras
        ct1.start()
        ct2.start()

        # ------------------------------
        # set up angles 
        # ------------------------------

        # cameras are the same, so only 1 needed
        angler = Frame_Angles(pixel_width,pixel_height,angle_width,angle_height)
        angler.build_frame()

        # ------------------------------
        # set up motion detection 
        # ------------------------------

        # motion camera1
        # using default detect values
        targeter1 = Frame_Motion()
        targeter1.contour_min_area = 1
        targeter1.targets_max = 1
        targeter1.target_on_contour = True # False = use box size
        targeter1.target_return_box = True # (x,y,bx,by,bw,bh)
        targeter1.target_return_size = True # (x,y,%frame)
        targeter1.targets_draw = True

        # motion camera2
        # using default detect values
        targeter2 = Frame_Motion()
        targeter2.contour_min_area = 1
        targeter2.targets_max = 1
        targeter2.target_on_contour = True # False = use box size
        targeter2.target_return_box = True # (x,y,bx,by,bw,bh)
        targeter2.target_return_size = True # (x,y,%frame)
        targeter2.targets_draw = True

        # ------------------------------
        # stabilize 
        # ------------------------------

        # pause to stabilize
        time.sleep(0.0) #modify according to lag

        # ------------------------------
        # targeting loop 
        # ------------------------------

        # variables
        maxsd = 10 # maximum size difference of targets, percent of frame
        

        # last positive target
        # from camera baseline midpoint
        X,Y,Z,D = 0,0,0,0

        #Frame dimensions

        # loop
        while 1:

            # get frames
            found1, frame1,b_box1 = ct1.next(black=True,wait=1)
            found2, frame2,b_box2 = ct2.next(black=True,wait=1)
            if not (found1 or found2):
            	print("Target not found")
            else:
            	print("Target found")
            	

            frame1_pil = Image.fromarray(frame1)
            w, h = frame1_pil.size
            np.copyto(frame1,np.array(frame1_pil))

            #w,h,d = np.shape(frame1)
            #print("frame dimensions",wid,hei,w,h)
            #multiply because the ssd returns normailized coordinates from [0,1]
            bxmin1 = b_box1[1]*w
            bymin1 = b_box1[0]*h
            bxmax1 = b_box1[3]*w
            bymax1 = b_box1[2]*h
            
            bxmin2 = b_box2[1]*w
            bymin2 = b_box2[0]*h
            bxmax2 = b_box2[3]*w
            bymax2 = b_box2[2]*h

            width,height,depth = np.shape(frame1)
            area1 = width*height
            
            width,height,depth = np.shape(frame2)
            area2 = width*height
            
            # targets
            targets1 = []
            targets2 = []
            
            bw1=abs(bxmax1-bxmin1)
            bh1=abs(bymax1-bymin1)

            bw2=abs(bxmax2-bxmin2)
            bh2=abs(bymax2-bymin2)
             # the bounding box coordinates are gotten from the ssd code
            ba1 = bw1*bh1
            ba2 = bw2*bh2

            p1 = 100*ba1/area1
            p2 = 100*ba2/area2
        
            tx1 = (bxmin1+bxmax1)/2
            ty1 = (bymin1+bymax1)/2
            targets1.append((tx1,ty1,p1))
            #cv2.circle(frame1,(int(bxmin1),int(bymin1)),15,(0,0,0),1)

            tx2 = (bxmin2+bxmax2)/2
            ty2 = (bymin2+bymax2)/2
            targets2.append((tx2,ty2,p2))
            #cv2.circle(frame2,(int(bxmin2),int(bymin2)),15,(0,0,0),1)
            # motion detection targets
            # targets1 = targeter1.targets(frame1,bx1,by1,bw1,bh1) # returns the centeroid and bounding box coordinates
            # targets2 = targeter2.targets(frame2,bx2,by2,bw2,bh2)

            #print("First box: ",targets1,"Second box: ",targets2)

            # check 1: motion in both frames
            if not (targets1 and targets2):
                print("No target found")
                x1k,y1k,x2k,y2k = [],[],[],[] # reset # if object not detected in the frames
            else:

                # split
                x1,y1,s1 = targets1[0]     #assuming there is only one face
                x2,y2,s2 = targets2[0]

                # check 2: similar size
                #if 100*(abs(s1-s2)/max(s1,s2)) > minsd:
                if abs(s1-s2) > maxsd:
                    x1k,y1k,x2k,y2k = [],[],[],[] # reset # if percent ckhange is more, they are two diff objects
                else:
                    #print("centroid means",x1,y1,x2,y2)
                    # get angles from camera centers
                    xlangle,ylangle = angler.angles_from_center(x1,y1,top_left=True,degrees=True)
                    xrangle,yrangle = angler.angles_from_center(x2,y2,top_left=True,degrees=True)
                    
                    # triangulate
                    X,Y,Z,D = angler.location(camera_separation,(xlangle,ylangle),(xrangle,yrangle),center=True,degrees=True)
                    
                                
                        
        
            # display camera centers
            angler.frame_add_crosshairs(frame1)
            angler.frame_add_crosshairs(frame2)

            # display coordinate data
            fps1 = int(ct1.current_frame_rate)
            fps2 = int(ct2.current_frame_rate)
            text = 'Y: {:3.1f}\nZ: {:3.1f}\nFPS: {}/{}'.format(Y,abs(Z),fps1,fps2)
            lineloc = 0
            lineheight = 30
            #print("FPS calculated","Depth: ",D)
            for t in text.split('\n'):
                lineloc += lineheight
                cv2.putText(frame1,
                            t,
                            (10,lineloc), # location
                            cv2.FONT_HERSHEY_PLAIN, # font
                            #cv2.FONT_HERSHEY_SIMPLEX, # font
                            1.5, # size
                            (0,0,0), # color
                            1, # line width
                            cv2.LINE_AA, #
                            False) #

            
            # display current target
            targeter1.frame_add_crosshairs(frame1,x1,y1,48)            
            targeter2.frame_add_crosshairs(frame2,x2,y2,48) 


            # display frame
            cv2.imshow("Left Camera 1",frame1)
            cv2.imshow("Right Camera 2",frame2)

            # detect keys
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty('Left Camera 1',cv2.WND_PROP_VISIBLE) < 1:
                break
            elif cv2.getWindowProperty('Right Camera 2',cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('q'):
                break
            elif key != 255:
                print('KEY PRESS:',[chr(key)])

    # ------------------------------
    # full error catch 
    # ------------------------------
    except:
        print(traceback.format_exc())

    # ------------------------------
    # close all
    # ------------------------------

    # close camera1
    try:
        ct1.stop()
    except:
        pass

    # close camera2
    try:
        ct2.stop()
    except:
        pass

    # kill frames
    cv2.destroyAllWindows()

    # done
    print('DONE')

# ------------------------------
# Camera Tread
# ------------------------------

class Camera_Thread:

    # IMPORTANT: a queue is much more efficient than a deque
    # the queue version runs at 35% of 1 processor
    # the deque version ran at 108% of 1 processor

    # ------------------------------
    # User Instructions
    # ------------------------------

    # Using the user variables (see below):
    # Set the camera source number (default is camera 0).
    # Set the camera pixel width and height (default is 1280x720).
    # Set the target (max) frame rate (default is 30).
    # Set the number of frames to keep in the buffer (default is 4).
    # Set buffer_all variable: True = no frame loss, for reading files, don't read another frame until buffer allows
    #                          False = allows frame loss, for reading camera, just keep most recent frame reads

    # Start camera thread using self.start().

    # Get next frame in using self.next(black=True,wait=1).
    #    If black, the default frame value is a black frame.
    #    If not black, the default frame value is None.
    #    If timeout, wait up to timeout seconds for a frame to load into the buffer.
    #    If no frame is in the buffer, return the default frame value.

    # Stop the camera using self.stop()

    # ------------------------------
    # User Variables
    # ------------------------------

    # camera setup
    camera_source = 0
    camera_width = 1280
    camera_height = 720
    camera_frame_rate = 30
    camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    detection_graph.as_default()
    sess = tf.Session(graph=detection_graph)
   

    # buffer setup
    buffer_length = 5
    buffer_all = False

    # ------------------------------
    # System Variables
    # ------------------------------

    # camera
    camera = None
    camera_init = 0.5

    # buffer
    buffer = None

    # control states
    frame_grab_run = False
    frame_grab_on = False

    # counts and amounts
    frame_count = 0
    frames_returned = 0
    current_frame_rate = 0
    loop_start_time = 0

    # ------------------------------
    # Functions
    # ------------------------------

    def start(self):

        # buffer
        if self.buffer_all:
            self.buffer = queue.Queue(self.buffer_length)
        else:
            # last frame only
            self.buffer = queue.Queue(1)

        # camera setup
        self.camera = cv2.VideoCapture(self.camera_source)
        self.camera.set(3,self.camera_width)
        self.camera.set(4,self.camera_height)
        self.camera.set(5,self.camera_frame_rate)
        self.camera.set(6,self.camera_fourcc)
        time.sleep(self.camera_init)

        # camera image vars
        self.camera_width  = int(self.camera.get(3))
        self.camera_height = int(self.camera.get(4))
        self.camera_frame_rate = int(self.camera.get(5))
        self.camera_mode = int(self.camera.get(6))
        self.camera_area = self.camera_width*self.camera_height

        # black frame (filler)
        self.black_frame = np.zeros((self.camera_height,self.camera_width,3),np.uint8)

        # set run state
        self.frame_grab_run = True
        
        # start thread
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()


    def stop(self):

        # set loop kill state
        self.frame_grab_run = False
        
        # let loop stop
        while self.frame_grab_on:
            time.sleep(0.1)

        # stop camera if not already stopped
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
        self.camera = None

        # drop buffer
        self.buffer = None


    def loop(self):

        # load start frame
        frame = self.black_frame
        if not self.buffer.full():
            self.buffer.put(frame,False)

        # status
        self.frame_grab_on = True
        self.loop_start_time = time.time()

        # frame rate
        fc = 0
        t1 = time.time()
        
        while 1:

            # external shut down
            if not self.frame_grab_run:
                break

            # true buffered mode (for files, no loss)
            if self.buffer_all:

                # buffer is full, pause and loop
                if self.buffer.full():
                    time.sleep(1/self.camera_frame_rate)

                # or load buffer with next frame
                else:
                    print("Inside True buffer_all condition")
                    grabbed,frame = self.camera.read()

                    

                    if not grabbed:
                        break

                    self.buffer.put(frame,False)
                    self.frame_count += 1
                    fc += 1

            # false buffered mode (for camera, loss allowed)
            else:
                
                grabbed,frame = self.camera.read()
                
                #if self.camera_source == 0: 
                #   while turn != 0:
                #      continue 
                #   cv2.imshow('object detection_0', cv2.resize(frame, (800,600)))
                #   turn = 2
                #else:
                #   while turn != 2:
                #      continue
                #   cv2.imshow('object detection_2', cv2.resize(frame, (800,600)))
                #   turn = 0
                if not grabbed:
                    break
                
                # open a spot in the buffer
                if self.buffer.full():
                    self.buffer.get()

                self.buffer.put(frame,False)
                self.frame_count += 1
                fc += 1
            
            # update frame read rate
            if fc >= 10:
                self.current_frame_rate = round(fc/(time.time()-t1),2)
                fc = 0
                t1 = time.time()
        
            #time.sleep(0.5)
               
       
                

        # shut down
        self.loop_start_time = 0
        self.frame_grab_on = False
        self.stop()

    def next(self,black=True,wait=0):

        # black frame default
        if black:
            frame = self.black_frame

        # no frame default
        else:
            frame = None

        # get from buffer (fail if empty)
        try:
            print("Reading from buffer")
            frame = self.buffer.get(timeout=wait)
            self.frames_returned += 1
            print("Got frame no",self.frames_returned,self.camera_source)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(frame, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = self.sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int64), #changed it from int32
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            
            np_boxes = np.squeeze(boxes)
            np_classes = np.squeeze(classes)
            index = 0 # Variable for getting box index, initially not a valid index
            found = False
            print("outside")
            for i in range(len(np_classes)): #traverse the length of np_classes
              if np_classes[i] == get_id_for_label("Human face"): # id of Human face
                print("original",np_boxes[i])
                index = i
                found = True


            # Prints the bounding box coordinate but not the frame.
            cv2.imshow('camera', cv2.resize(frame, (800,600)))
        except queue.Empty:
            print('Queue Empty!')
            #print(traceback.format_exc())
            pass

        # done
        
        return found, frame, np_boxes[index]

# ------------------------------
# Motion Detection
# ------------------------------

class Frame_Motion:

    # ------------------------------
    # User Instructions
    # ------------------------------

    # ------------------------------
    # User Variables
    # ------------------------------

    # contour size
    contour_min_area = 1  # percent of frame area
    contour_max_area = 100 # percent of frame area

    # target select
    targets_max = 1 # max targets returned, hence will not detect more than 4 things in a frame
    target_on_contour = True # else use box size
    target_return_box  = False # True = return (x,y,bx,by,bw,bh), else check target_return_size
    target_return_size = True # True = return (x,y,percent_frame_size), else just (x,y)

    # display targets
    targets_draw  = True
    targets_point = 4 # centroid radius
    targets_pline = -1 # border width
    targets_color = (0,0,255) # BGR color

    # ------------------------------
    # System Variables
    # ------------------------------

    # ------------------------------
    # Functions
    # ------------------------------

    def targets(self,frame,bx,by,bw,bh): # This has to be passed from the ssd code (this function is called there)

        # frame dimensions
        width,height,depth = np.shape(frame)
        area = width*height

       
        # targets
        targets = []
        
         # the bounding box coordinates are gotten from the ssd code
        ba = bw*bh

        p = 100*ba/area
        if (p >= self.contour_min_area) and (p <= self.contour_max_area):
            tx = bx+int(bw/2)
            ty = by+int(bh/2)
            targets.append((p,tx,ty,bx,by,bw,bh))

            
        
        # add targets to frame
        if self.targets_draw:
            for size,x,y,bx,by,bw,bh in targets:
                cv2.circle(frame,(x,y),self.targets_point,self.targets_color,self.targets_pline)

       

        # return target x,y
        if self.target_return_box:
            return [(x,y,bx,by,bw,bh) for (size,x,y,bx,by,bw,bh) in targets]
        elif self.target_return_size:
            return [(x,y,size) for (size,x,y,bx,by,bw,bh) in targets]
        else:
            return [(x,y) for (size,x,y,bx,by,bw,bh) in targets]

    def frame_add_crosshairs(self,frame,x,y,r=20,lc=(0,0,255),cc=(0,0,255),lw=1,cw=1):

        x = int(round(x,0))
        y = int(round(y,0))
        r = int(round(r,0))

        cv2.line(frame,(x,y-r*2),(x,y+r*2),lc,lw)
        cv2.line(frame,(x-r*2,y),(x+r*2,y),lc,lw)

        cv2.circle(frame,(x,y),r,cc,cw)



# ------------------------------
# Frame Angles and Distance
# ------------------------------

class Frame_Angles:

    # ------------------------------
    # User Instructions
    # ------------------------------

    # Set the pixel width and height.
    # Set the angle width (and angle height if it is disproportional).
    # These can be set during init, or afterwards.

    # Run build_frame.

    # Use angles_from_center(self,x,y,top_left=True,degrees=True) to get x,y angles from center.
    # If top_left is True, input x,y pixels are measured from the top left of frame.
    # If top_left is False, input x,y pixels are measured from the center of the frame.
    # If degrees is True, returned angles are in degrees, otherwise radians.
    # The returned x,y angles are always from the frame center, negative is left,down and positive is right,up.

    # Use pixels_from_center(self,x,y,degrees=True) to convert angle x,y to pixel x,y (always from center).
    # This is the reverse of angles_from_center.
    # If degrees is True, input x,y should be in degrees, otherwise radians.

    # Use frame_add_crosshairs(frame) to add crosshairs to a frame.
    # Use frame_add_degrees(frame) to add 10 degree lines to a frame (matches target).
    # Use frame_make_target(openfile=True) to make an SVG image target and open it (matches frame with degrees).

    # ------------------------------
    # User Variables
    # ------------------------------

    pixel_width = 1280
    pixel_height = 720

    angle_width = 46
    angle_height = 36
    
    # ------------------------------
    # System Variables
    # ------------------------------

    x_origin = None
    y_origin = None

    x_adjacent = None
    x_adjacent = None

    # ------------------------------
    # Init Functions
    # ------------------------------

    def __init__(self,pixel_width=None,pixel_height=None,angle_width=None,angle_height=None):

        # full frame dimensions in pixels
        if type(pixel_width) in (int,float):
            self.pixel_width = int(pixel_width)
        if type(pixel_height) in (int,float):
            self.pixel_height = int(pixel_height)

        # full frame dimensions in degrees
        if type(angle_width) in (int,float):
            self.angle_width = float(angle_width)
        if type(angle_height) in (int,float):
            self.angle_height = float(angle_height)

        # do initial setup
        self.build_frame()

    def build_frame(self):

        # this assumes correct values for pixel_width, pixel_height, and angle_width

        # fix angle height
        if not self.angle_height:
            self.angle_height = self.angle_width*(self.pixel_height/self.pixel_width)

        # center point (also max pixel distance from origin)
        self.x_origin = int(self.pixel_width/2)
        self.y_origin = int(self.pixel_height/2) #added negative sign

        # theoretical distance in pixels from camera to frame
        # this is the adjacent-side length in tangent calculations
        # the pixel x,y inputs is the opposite-side lengths
        self.x_adjacent = self.x_origin / math.tan(math.radians(self.angle_width/2))
        self.y_adjacent = self.y_origin / math.tan(math.radians(self.angle_height/2))

    # ------------------------------
    # Pixels-to-Angles Functions
    # ------------------------------

    def angles(self,x,y):

        return self.angles_from_center(x,y)

    def angles_from_center(self,x,y,top_left=True,degrees=True):

        # x = pixels right from left edge of frame
        # y = pixels down from top edge of frame
        # if not top_left, assume x,y are from frame center
        # if not degrees, return radians

        if top_left:
            x = x - self.x_origin
            y = -self.y_origin - y

        xtan = x/self.x_adjacent
        ytan = y/self.y_adjacent

        xrad = math.atan(xtan)
        yrad = math.atan(ytan)

        if not degrees:
            return xrad,yrad

        return math.degrees(xrad),math.degrees(yrad)

    def pixels_from_center(self,x,y,degrees=True):

        # this is the reverse of angles_from_center

        # x = horizontal angle from center
        # y = vertical angle from center
        # if not degrees, angles are radians

        if degrees:
            x = math.radians(x)
            y = math.radians(y)

        return int(self.x_adjacent*math.tan(x)),int(self.y_adjacent*math.tan(y))

    # ------------------------------
    # 3D Functions
    # ------------------------------

    def distance(self,*coordinates):
        return self.distance_from_origin(*coordinates)

    def distance_from_origin(self,*coordinates):
        return math.sqrt(sum([x**2 for x in coordinates]))
    
    def intersection(self,pdistance,langle,rangle,degrees=False):

        # return (X,Y) of target from left-camera-center

        # pdistance is the measure from left-camera-center to right-camera-center (point-to-point, or point distance)
        # langle is the left-camera  angle to object measured from center frame (up/right positive)
        # rangle is the right-camera angle to object measured from center frame (up/right positive)
        # left-camera-center is origin (0,0) for return (X,Y)
        # X is measured along the baseline from left-camera-center to right-camera-center
        # Y is measured from the baseline

        # fix degrees
        if degrees:
            langle = math.radians(langle)
            rangle = math.radians(rangle)

        # fix angle orientation (from center frame)
        # here langle is measured from right baseline
        # here rangle is measured from left  baseline
        langle = math.pi/2 - langle
        rangle = math.pi/2 + rangle

        # all calculations using tangent
        ltan = math.tan(langle)
        rtan = math.tan(rangle)

        # get Y value
        # use the idea that pdistance = ( Y/ltan + Y/rtan )
        Y = pdistance / ( 1/ltan + 1/rtan )

        # get X measure from left-camera-center using Y
        X = Y/ltan

        # done
        return X,Y

    def location(self,pdistance,lcamera,rcamera,center=False,degrees=True):

        # return (X,Y,Z,D) of target from left-camera-center (or baseline midpoint if center-True)

        # pdistance is the measure from left-camera-center to right-camera-center (point-to-point, or point distance)
        # lcamera = left-camera-center (Xangle-to-target,Yangle-to-target)
        # rcamera = right-camera-center (Xangle-to-target,Yangle-to-target)
        # left-camera-center is origin (0,0) for return (X,Y)
        # X is measured along the baseline from left-camera-center to right-camera-center
        # Y is measured from the baseline
        # Z is measured vertically from left-camera-center (should be same as right-camera-center)
        # D is distance from left-camera-center (based on pdistance units)

        # separate values
        lxangle,lyangle = lcamera
        rxangle,ryangle = rcamera

        # yangle should be the same for both cameras (if aligned correctly)
        yangle = (lyangle+ryangle)/2

        # fix degrees
        if degrees:
            lxangle = math.radians(lxangle)
            rxangle = math.radians(rxangle)
            yangle  = math.radians( yangle)

        # get X,Z (remember Y for the intersection is Z frame)
        X,Z = self.intersection(pdistance,lxangle,rxangle,degrees=False)

        # get Y
        # using yangle and 2D distance to target
        Y = math.tan(yangle) * self.distance_from_origin(X,Z)

        # baseline-center instead of left-camera-center
        if center:
            X -= pdistance/2

        # get 3D distance
        D = self.distance_from_origin(X,Y,Z)

        # done
        return X,Y,Z,D
    
    # ------------------------------
    # Tertiary Functions
    # ------------------------------

    def frame_add_crosshairs(self,frame):

        # add crosshairs to frame to aid in aligning

        cv2.line(frame,(0,self.y_origin),(self.pixel_width,self.y_origin),(0,255,0),1)
        cv2.line(frame,(self.x_origin,0),(self.x_origin,self.pixel_height),(0,255,0),1)

        cv2.circle(frame,(self.x_origin,self.y_origin),int(round(self.y_origin/8,0)),(0,255,0),1)

    def frame_add_degrees(self,frame):

        # add lines to frame every 10 degrees (horizontally and vertically)
        # use this to test that your angle values are set up properly

        for angle in range(10,95,10):

            # calculate pixel offsets
            x,y = self.pixels_from_center(angle,angle)

            # draw verticals
            if x <= self.x_origin:
                cv2.line(frame,(self.x_origin-x,0),(self.x_origin-x,self.pixel_height),(255,0,255),1)
                cv2.line(frame,(self.x_origin+x,0),(self.x_origin+x,self.pixel_height),(255,0,255),1)

            # draw horizontals
            if y <= self.y_origin:
                cv2.line(frame,(0,self.y_origin-y),(self.pixel_width,self.y_origin-y),(255,0,255),1)
                cv2.line(frame,(0,self.y_origin+y),(self.pixel_width,self.y_origin+y),(255,0,255),1)

    def frame_make_target(self,outfilename='targeting_angles_frame_target.svg',openfile=False):

        # this will make a printable target that matches the frame_add_degrees output
        # use this to test that your angle values are set up properly
        
        # svg size
        ratio = self.pixel_height/self.pixel_width
        width = 1600
        height = 1600 * ratio

        #svg frame locations
        x_origin = width/2
        y_origin = height/2
        distance = width*0.5

        # start svg
        svg  = '<svg xmlns="http://www.w3.org/2000/svg"\n'
        svg += 'xmlns:xlink="http://www.w3.org/1999/xlink"\n'
        svg += 'width="{}px"\n'.format(width)
        svg += 'height="{}px">\n'.format(height)

        # crosshairs
        svg += '<line x1="{}" x2="{}" y1="{}" y2="{}" stroke-width="1" stroke="green"/>\n'.format(0,width,y_origin,y_origin)
        svg += '<line x1="{}" x2="{}" y1="{}" y2="{}" stroke-width="1" stroke="green"/>\n'.format(x_origin,x_origin,0,height)

        # center circle
        svg += '<circle cx="{}" cy="{}" r="{}" stroke="green" stroke-width="1" fill="none"/>'.format(x_origin,y_origin,y_origin/8)

        # distance from screen line
        svg += '<line x1="{0}" x2="{1}" y1="{2}" y2="{2}" stroke-width="1" stroke="red"/>\n'.format(x_origin-distance/2,x_origin+distance/2,y_origin-y_origin/8)
        svg += '<line x1="{0}" x2="{0}" y1="{1}" y2="{2}" stroke-width="1" stroke="red"/>\n'.format(x_origin-distance/2,y_origin-y_origin/16,y_origin-y_origin/8)
        svg += '<line x1="{0}" x2="{0}" y1="{1}" y2="{2}" stroke-width="1" stroke="red"/>\n'.format(x_origin+distance/2,y_origin-y_origin/16,y_origin-y_origin/8)

        # add degree lines
        for angle in range(10,95,10):
            pixels = distance * math.tan(math.radians(angle))

            # draw verticals
            if pixels <= x_origin:
                svg += '<line x1="{0}" x2="{0}" y1="0" y2="{1}" stroke-width="1" stroke="black"/>\n'.format(x_origin-pixels,height)
                svg += '<line x1="{0}" x2="{0}" y1="0" y2="{1}" stroke-width="1" stroke="black"/>\n'.format(x_origin+pixels,height)

            # draw horizontals
            if pixels <= y_origin:
                svg += '<line x1="0" x2="{0}" y1="{1}" y2="{1}" stroke-width="1" stroke="black"/>\n'.format(width,y_origin-pixels)
                svg += '<line x1="0" x2="{0}" y1="{1}" y2="{1}" stroke-width="1" stroke="black"/>\n'.format(width,y_origin+pixels)

        # end svg
        svg += '</svg>'

        # write file
        outfile = open(outfilename,'w')
        outfile.write(svg)
        outfile.close()

        # open file
        if openfile:
            import webbrowser
            webbrowser.open(os.path.abspath(outfilename))

# ------------------------------
# Testing
# ------------------------------

if __name__ == '__main__':
    run()

# ------------------------------
# End
# ------------------------------
    

#targeting_tools.py
#Displaying targeting_tools.py.
