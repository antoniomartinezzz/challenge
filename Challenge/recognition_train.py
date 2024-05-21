import matplotlib.pyplot as plt
import os
import glob
import tensorflow
from deepface import DeepFace
import pickle
from tqdm import tqdm 
import cv2
from sklearn.ensemble import RandomForestClassifier


train_image_paths = glob.glob("./data/*/**.jpg") #Path list for all images in the database.

#'Training' -> calculate all embedding vectors for faces in the database, using one pretrained face detection model. 

train_encodings = [] #list contains face encodings for employees
names = [] #contains names for corresponding face in i-th position. 

#METHODS -> different methods for faxe recognition or different methods to obtain closest face in database. 

for image_path in tqdm(train_image_paths): 
    name = image_path.split('/')[-2] #extract name of employee from image path
    embedding_objs = DeepFace.represent(img_path= image_path, model_name='Facenet', normalization="Facenet", detector_backend='mtcnn' ) #calculate embedding for detected face in image
    assert len(embedding_objs)==1, f"{len(embedding_objs)} faces were detected for {image_path.split('/')[-1]} , 1 expected"
    #Add embedding and corresponding employee name to embedding list. 
    train_encodings.append(embedding_objs[0]['embedding']) 
    names.append(name)

#Create dictionary with train encodings for NN
train_data = {'embeddings': train_encodings, 'names': names}

#save file for later use
f = open("face_enc", "wb")
f.write(pickle.dumps(train_data))
f.close()




    



