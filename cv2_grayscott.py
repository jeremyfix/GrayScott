import cv2
import sys
import random

import grayscott


cv2.namedWindow('u')
cv2.namedWindow('v')

key = 0
run = False

mode = int(sys.argv[1])
N = 128
model = grayscott.Model('solitons', N=N, mode=mode, measure_fps=True)
model.init()


while key != ord('q'):
	if(run): 
		model.step()	
	cv2.imshow('u', model.ut)
	cv2.imshow('v', model.vt)

	key = cv2.waitKey(1) & 0xFF

	if(key == ord('c')):
		c = (random.randint(0, N-1), random.randint(0, N-1))
		cv2.circle(model.vt_1, c , N/4, 0, -1)
	elif key == ord('s'):
		run = not run
		print("Running ? : " + str(run))
	elif key == ord('i'):
		model.init()

