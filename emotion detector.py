#Emotion Detector Test

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import pickle
import numpy as np

import cv2

c = 0

try:
    with open('model.bin','rb') as mdl:
        model = pickle.load(mdl)
        print("\n\nModel Loaded\n\n")
        
except:
    train_path = 'train'
    valid_path = 'test'


    train_datagen = ImageDataGenerator(rescale=1./48)
    valid_datagen = ImageDataGenerator(rescale=1./48)

    train_batches = train_datagen.flow_from_directory(train_path, target_size=(47,47), classes=['sad','neutral','happy'], batch_size=32)
    valid_batches = valid_datagen.flow_from_directory(valid_path, target_size=(47,47), classes=['sad','neutral','happy'], batch_size=32)


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(47,47,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(3, activation='softmax')
    ])


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    model.fit(train_batches, validation_data=valid_batches, epochs=15)

    with open('model.bin','wb') as mdl:
        pickle.dump(model,mdl)

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('path\\to\\the\\file\\haarcascade_frontalface_alt2.xml')

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        x1 = frame[y:y+h,x:x+w]
        x1 = cv2.resize(x1,(47,47))
        x1 = x1/48.0
        x1 = np.expand_dims(x1,axis=0)

        font = cv2.FONT_HERSHEY_SIMPLEX

        org = (x,y)

        fontScale = 1

        color = (255, 0, 0)

        thickness = 2


        prediction = model.predict(x1)

        if(max(prediction[0]) == prediction[0][0]):
            frame = cv2.putText(frame, 'Sad', org, font, fontScale, color, thickness, cv2.LINE_AA)
        elif(max(prediction[0]) == prediction[0][1]):
            frame = cv2.putText(frame, 'Neutral', org, font, fontScale, color, thickness, cv2.LINE_AA)
        else:
            frame = cv2.putText(frame, 'Happy', org, font, fontScale, color, thickness, cv2.LINE_AA)


    cv2.imshow('frame', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
	    
cap.release()

cv2.destroyAllWindows()

    
