#!/usr/bin/env python
# coding: utf-8

# # INTERACTIVE WINDOW

# In[1]:


import ipywidgets as widgets
from tkinter import *
from tkinter import ttk


# In[2]:


root = Tk()
root.title('Facial Emotion Tracker')
root.geometry("600x400")

def change_bar_len(new_val):
    bar['value'] = new_val

def quit():
    root.quit()

bar = ttk.Progressbar(root, orient=HORIZONTAL, length=300, mode='determinate')
bar.pack(pady=20)

btn = Button(root, text="Quit", command=root.destroy)
btn.pack(pady=20)

root.mainloop()


# In[ ]:





# # TAKING SCREENSHOT

# In[3]:


import time
import pyscreenshot as ImageGrab
import schedule
from datetime import datetime


# In[ ]:


def take_ss():
    
    print('Taking screenshot...')
    
    image_name = f"zoom-ss-{str(datetime.now())}" # naming image
    ss = ImageGrab.grab() # taking ss
    
    filepath = f"./FACE-SCREENSHOTS-2/{image_name}.png" # setting directory as "Face Screenshots"
    ss.save(filepath) # saving the image
    
    print('Screenshot taken...')
    
    return filepath


# In[ ]:


def main():
    schedule.every(5).seconds.do(take_ss) # take ss every 5 seconds
    
    while True:
        schedule.run_pending()
        time.sleep(1)


# In[ ]:


if __name__ == '__main__':
    main()


# In[ ]:





# # TESTING EMOTION FROM PHOTO

# In[4]:


import cv2
import pyautogui
from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np


# In[5]:


img = cv2.imread('FACE-SCREENSHOTS-2/test-1.png')
# img = cv2.imread('FACE-SCREENSHOTS-2/dub-me.png')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[293]:


plt.imshow(img) #BGR


# In[294]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faces = faceCascade.detectMultiScale(gray, 1.1, 4)

for(x, y, w, h) in faces:
    cropped_image = img[y:y+h, x:x+w]
    cv2.imshow('image',cropped_image)
    print(x)


# In[295]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faces = faceCascade.detectMultiScale(gray, 1.1, 4)

for(x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    
    try:
        analyze = DeepFace.analyze(frame,actions=['emotions'])  #same thing is happing here as the previous example, we are using the analyze class from deepface and using ‘frame’ as input
        print(analyze['dominant_emotion'])  #here we will only go print out the dominant emotion also explained in the previous example
    except:
          print("no face")
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    
    
    


# In[298]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faces = faceCascade.detectMultiScale(gray, 1.1, 4)

names = []
emotion_preds = []

counter = 0

for(x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    cropped_image = img[y:y+h, x:x+w]

    #plt.imshow('image',img)
    
    predictions = DeepFace.analyze(cropped_image)
    print(predictions['dominant_emotion'])
    
    emotion_preds.append(predictions['dominant_emotion'])
    
    
    


# In[299]:


emotion_preds


# In[300]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[301]:


img = cv2.imread('FACE-SCREENSHOTS-2/test-1.png')
cropped_image = img[y:y+h, x:x+w]
plt.imshow(cropped_image)


# In[302]:


for(x, y, w, h) in faces:
    cropped_image = img[y:y+h, x:x+w]
    plt.imshow(cropped_image)


# In[303]:


for(x, y, w, h) in faces:
    cropped_image = img[y:y+h, x:x+w]
    plt.imshow(cropped_image)
    
    predictions = DeepFace.analyze(cropped_image)
    print(predictions['dominant_emotion'])


# In[304]:


predictions = DeepFace.analyze(img)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, predictions['dominant_emotion'], (0,200), font, 10, (0,0,255), 2, cv2.LINE_4)


# In[305]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[ ]:





# # GRABBING NAMES

# In[206]:


img


# In[257]:


width, height = img.size

area_1 = (0, height/5, width, height-(height/5))
crop_img_1 = img.crop(area_1)

crop_img_1.show()


# In[263]:


area_2 = (width/2, height/5, width*(3/4), height-(height/6))
crop_img_2 = img.crop(area_2)

crop_img_2.show()


# In[271]:


from PIL import Image
from pytesseract import pytesseract
import io as os

text_1 = pytesseract.image_to_string(crop_img_1, config='--psm 3')
text_2 = pytesseract.image_to_string(crop_img_2, config='--psm 3')

print(text_1)
print(text_2)

names.append(text_1)
names.append(text_2)


# In[272]:


names


# In[ ]:




