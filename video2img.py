import cv2
import sys
import progressbar
vidcap = cv2.VideoCapture(sys.argv[1])
success,image = vidcap.read()
count = 0
success = True
while success:
    success, image = vidcap.read()
    print 'Read a new frame: ', success
    cv2.imwrite(sys.argv[2] + "/%d.jpg" % count, image)     # save frame as JPEG file
    print "frame", count
    count += 1
