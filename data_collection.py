#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data


# In[2]:


import cv2
import numpy as np

#Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip=0;
dataset_path = '/home/mohit123/Machlea/facrecog/'
face_data = [] 
file_name = input("Enter the name of the person : ")
while True:
    ret,frame = cap.read()
    if ret == False:
        continue
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame,1.3,5)
    #print(faces)
    faces = sorted(faces,key=lambda f:f[2]*f[3])#sorts the faces on the basis of their size  
    
    
    # Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
    for face in faces[-1:]:#the largest face comes to the front of the list therefore -1:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
        #Extract (Crop out the required face) : Region of Interest(format is frame[y value,x value])
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))# 100X100 is the size of the window
        
        skip+=1;
        if skip%10==0:#capture every tenth pic
            face_data.append(face_section)
            print(len(face_data))
    cv2.imshow("Faces",frame)
    cv2.imshow("Face_section",face_section)
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
        
#convert our face list into numpy array
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully save at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()


# 
#   

# In[ ]:




