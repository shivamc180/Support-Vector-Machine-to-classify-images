import os
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from PIL import Image


dir = r'I:\\coding\\projects\\personal_ws\\Internship\\Prodigy InfoTech Internship\\Support Vector Machines\\Dataset\\dogs-vs-cats'

categories = ['Cat', 'Dog']

data = []


for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)
    
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        pet_img = cv2.imread(imgpath, 0)    
        try:
            pet_img = cv2.resize(pet_img, (50,50))
            image = np.array(pet_img).flatten()
        
        
            data.append([image, label])
            
        except Exception as e:
            pass
        
    
print(len(data))
                

