#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 1. load the training data (numpy arrays of all the persons)
        # x- values are stored in the numpy arrays
        # y-values we need to assign for each person
# 2. Read a video stream using opencv
# 3. extract faces out of it
# 4. use knn to find the prediction of face (int)
# 5. map the predicted id to name of the user 
# 6. Display the predictions on the screen - bounding box and name


# In[3]:


import cv2
import numpy as np 
import os 


# In[4]:


#KNN
def distance(v1, v2):
    # Eucledian 
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
    dist = []
    
    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]
    
    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]


# In[5]:


#Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip=0;
dataset_path = '/home/mohit123/Machlea/facrecog/'
face_data = [] 
label=[] #will form the y values of the data

class_id=0#starting id(labels for the given file)
names={} #mapping b/w id and name

# Data Preparation
for fx in os.listdir(dataset_path): #os.listdir gives all the files in the data folder
    if fx.endswith('.npy'):
        #Create a mapping btw class_id and name
        names[class_id] = fx[:-4]# -4 so as to remove .npy
        
    
        print("Loaded "+fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        #Create Labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        label.append(target)
        
face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(label,axis=0).reshape((-1,1))

#print(face_dataset.shape)
#print(face_labels.shape)


# In[6]:


trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)


# In[7]:


#testing
while True:
    ret,frame = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame,1.3,5)
    if(len(faces)==0):
        continue
    for face in faces:
        x,y,w,h = face

        #Get the face ROI
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        #Predicted Label (out)
        out = knn(trainset,face_section.flatten())
        
        
        #Display on the screen the name and rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)#last parameter so as to get a better look
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
    cv2.imshow("Faces",frame)
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




