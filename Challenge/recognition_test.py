
from deepface import DeepFace
import pickle
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import math
from sklearn.ensemble import RandomForestClassifier

metric = 'cosine' #we use cosine similarity to comute similaroty between input image face embeddings and train embeddings.

def recognize_NN(imput_image_path: str):
    #We expect that a face pair of same person should be more similar than a face pair of different persons.
    train_data = pickle.loads(open('face_enc', "rb").read())

    #Compute embedding for all detected face in input image (we use diferent detector backend for speed)
    input_embeddings = DeepFace.represent(img_path= imput_image_path, normalization="Facenet", model_name='Facenet', detector_backend='dlib')
    objs = DeepFace.analyze(img_path = imput_image_path, actions = ['emotion'], detector_backend='dlib')

    image = cv2.imread(imput_image_path)
    for (i,detected_face) in enumerate(input_embeddings): 
        embedding = detected_face['embedding']
        x = detected_face['facial_area']['x']
        y = detected_face['facial_area']['y']
        w = detected_face['facial_area']['w']
        h = detected_face['facial_area']['h']

        #recognition using NN
        min = math.inf
        employee = 'Unknown'
        threshold = 0.4
        for (j,train_embedding) in enumerate(train_data['encodings']):
            cosine_distance = 1 - cosine_similarity([embedding],[train_embedding])
            if cosine_distance < min and cosine_distance < threshold:
                min = cosine_distance
                employee = train_data['names'][j]
        
        cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(image, employee, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        #Emotion
        cv2.putText(image, objs[i]['dominant_emotion'], (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imwrite('./test_result_NN/'+imput_image_path.split('/')[-1], image)

def recognize_RF(imput_image_path: str):

    train_data = pickle.loads(open('face_enc', "rb").read())
    clf = RandomForestClassifier(max_depth=40, random_state=0)
    clf.fit(train_data['encodings'], train_data['names'])
    input_embeddings = DeepFace.represent(img_path= imput_image_path, normalization="Facenet", model_name='Facenet', detector_backend='dlib')
    objs = DeepFace.analyze(img_path = imput_image_path, actions = ['emotion'], detector_backend='dlib')
    image = cv2.imread(imput_image_path)

    for (i,detected_face) in enumerate(input_embeddings): 
        embedding = detected_face['embedding']
        x = detected_face['facial_area']['x']
        y = detected_face['facial_area']['y']
        w = detected_face['facial_area']['w']
        h = detected_face['facial_area']['h']
        employee = clf.predict([embedding])

        cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(image, employee[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(image, objs[i]['dominant_emotion'], (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    cv2.imwrite('./test_result_RF/'+imput_image_path.split('/')[-1], image)


test_image = './test/IMG_8320.jpg'
recognize_NN(test_image)
recognize_RF(test_image)





