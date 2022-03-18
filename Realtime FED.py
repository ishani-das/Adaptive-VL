#!/usr/bin/env python
# coding: utf-8

# In[80]:


import cv2
import pyautogui


# In[81]:


from deepface import DeepFace


# In[82]:


img = cv2.imread('FED/Face-Screenshots/ss-1.png')


# In[83]:


import matplotlib.pyplot as plt


# In[84]:


plt.imshow(img) #BGR


# In[85]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[86]:


predictions = DeepFace.analyze(img)


# In[87]:


predictions


# In[88]:


type(predictions)


# In[89]:


predictions['dominant_emotion']


# # Rectangle

# In[119]:


# img = cv2.imread('FED/Face-Screenshots/ss-1.png')
img = cv2.imread('FED/FACE-SCREENSHOTS-2/test-1.png')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[120]:


plt.imshow(img) #BGR


# In[121]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faces = faceCascade.detectMultiScale(gray, 1.1, 4)

for(x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)


# In[122]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[105]:


font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(img, predictions['dominant_emotion'], (0,200), font, 1, (0,0,255), 2, cv2.LINE_4)


# In[106]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[96]:


predictions = DeepFace.analyze(img)
predictions


# # Integrating realtime video feed

# In[ ]:


import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cannot open webcam')
    
while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(frame, actions=['emotion'])
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, result['dominant_emotion'], (0,200), font, 3, (0,0,255), 2, cv2.LINE_4)
    cv2.imshow('Video Feed', frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
        
        


# In[ ]:




