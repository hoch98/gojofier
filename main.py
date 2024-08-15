
import cv2
from scipy import ndimage
import math
import numpy as np

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def output(path):

   img = cv2.imread(path)
   blindfold = cv2.imread("blindfold.png", -1)

   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   face_cascade = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')
   eye_cascade = cv2.CascadeClassifier('haarcascades\haarcascade_eye_tree_eyeglasses.xml')
   nose_cascade = cv2.CascadeClassifier('haarcascades\haarcascade_mcs_nose.xml')

   faces = face_cascade.detectMultiScale(gray, 1.3, 4)
   print('Number of detected faces:', len(faces))

   for (x,y,w,h) in faces:
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = img[y:y+h, x:x+w]

      midEye = (0, 0)
      nose = (0, 0)
      topNose = (0, 0)

      eyes = eye_cascade.detectMultiScale(roi_gray)
      noses = nose_cascade.detectMultiScale(roi_gray)
      
      for (ex,ey,ew,eh) in eyes:
         midEye = (midEye[0]+ex+ew/2, midEye[1]+ey+eh/2)
      
      midEye = (int(midEye[0]/2), int(midEye[1]/2))
      cv2.circle(roi_color, midEye, 10, (0, 0, 255), -1)

      for (ex,ey,ew,eh) in noses:
         nose = (int(ex+ew/2), int(ey+eh/2))
         topNose = (int(ex+ew/2), ey)

      noseHeight = topNose[1]-nose[1]
      angle = math.atan((midEye[1]-nose[1])/(midEye[0]-nose[0]))
      print(angle)
      realNose = (int(nose[0]-noseHeight*math.cos(angle)), int(nose[1]-noseHeight*math.sin(angle)))

      cv2.circle(roi_color, realNose, 10, (0, 0, 255), -1)

      dist = int(math.dist(realNose, midEye)*2)

      # blindfold = cv2.resize(blindfold, (w, dist), interpolation = cv2.INTER_LINEAR)
      
      # x_offset = x
      # y_offset = midEye[1]-int(dist/2)+y

      # blindfold = ndimage.rotate(blindfold, angle*180/math.pi)

      # y1, y2 = y_offset, y_offset + blindfold.shape[0]
      # x1, x2 = x_offset, x_offset + blindfold.shape[1]

      # alpha_s = blindfold[:, :, 3] / 255.0
      # alpha_l = 1.0 - alpha_s

      # for c in range(0, 3):
      #    img[y1:y2, x1:x2, c] = (alpha_s * blindfold[:, :, c] +
      #                               alpha_l * img[y1:y2, x1:x2, c])
      

   cv2.imwrite("output-"+path.split(".")[0]+".png", img)

output("test-subject3.png")