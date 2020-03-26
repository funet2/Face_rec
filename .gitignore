from keras.models import Model,Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout
import keras
#Initiliase
Classifier=Sequential()

#Create models
Classifier.add(Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)))
Classifier.add(MaxPooling2D(pool_size=(2,2)))

Classifier.add(Conv2D(32,(3,3),activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2,2)))

Classifier.add(Flatten())
Classifier.add(Dense(128,activation='relu'))

Classifier.add(Dropout(0.5))
Classifier.add(Dense(1,activation='sigmoid'))

#Compilation
Classifier.compile(loss='binary_crossentropy'
                    ,optimizer='adam'
                    ,metrics=['accuracy'])
 
batch_size=8

from keras.preprocessing.image import ImageDataGenerator

#Training
Train_Data=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
#Testing Datagen
Test_Data=ImageDataGenerator(rescale=1./255)

Train_Gen=Train_Data.flow_from_directory(
    './DataSet',
    target_size=(64,64),
    batch_size=batch_size,
    class_mode='binary',
    classes=['Moncef']
)
Test_Gen=Test_Data.flow_from_directory(
    './DataSet',
    target_size=(64,64),
    batch_size=batch_size,
    class_mode='binary',
    classes=['Save']
)
Classifier.fit_generator(
    Train_Gen,
    steps_per_epoch=1000,
    epochs=20,
    validation_data=Test_Gen,
    validation_steps=200
)

Classifier.save('MyImages.Model')
