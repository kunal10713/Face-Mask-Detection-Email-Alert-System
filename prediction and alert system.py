#!/usr/bin/env python
# coding: utf-8



from keras.models import load_model
import cv2
import numpy as np
import tkinter
from tkinter import messagebox
import smtplib

# Initialize Tkinter
root = tkinter.Tk()
root.withdraw()

#Load trained deep learning model
model = load_model('face_mask_detection_alert_system1.h5')

#Classifier to detect face
face_det_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture Video
vid_source=cv2.VideoCapture(0)

# Dictionaries containing details of Wearing Mask and Color of rectangle around face. If wearing mask then color would be 
# green and if not wearing mask then color of rectangle around face would be red
text_dict={0:'Mask ON',1:'No Mask'}
rect_color_dict={0:(0,255,0),1:(0,0,255)}

SUBJECT = "Subject"   
TEXT = "One Visitor violated Face Mask Policy. See in the camera to recognize user. A Person has been detected without a face mask. Please Alert the authorities."
 

# While Loop to continuously detect camera feed
while(True):

    ret, img = vid_source.read()
    grayscale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_det_classifier.detectMultiScale(grayscale_img,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img = grayscale_img[y:y+w,x:x+w]
        resized_img = cv2.resize(face_img,(112,112))
        normalized_img = resized_img/255.0
        reshaped_img = np.reshape(normalized_img,(1,112,112,1))
        result=model.predict(reshaped_img)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),rect_color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),rect_color_dict[label],-1)
        cv2.putText(img, text_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2) 
        
        # If label = 1 then it means wearing No Mask and 0 means wearing Mask
        if (label == 1):
            # Throw a Warning Message to tell user to wear a mask if not wearing one. This will stay
            #open and No Access will be given He/She wears the mask
            messagebox.showwarning("Warning","Access Denied. Please wear a Face Mask")
            
            # Send an email to the administrator if access denied/user not wearing face mask 
            message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT)
            mail = smtplib.SMTP('smtp.gmail.com', 587)
            mail.ehlo()
            mail.starttls()
            mail.login('[YOUR_EMAIL_ID], '[YOUR_PASSWORD] ')
            mail.sendmail('[YOUR_EMAIL_ID]','[YOUR_EMAIL_ID]',message)
            mail.close
        else:
            pass
            break

    cv2.imshow('LIVE Video Feed',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
vid_source.release()






