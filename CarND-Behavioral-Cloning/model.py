import csv
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.optimizers import Adam

data = [] 
batch_size = 32

with open('./data/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        data.append(line) 

train_samples, validation_samples = train_test_split(data, test_size = 0.20)

# Generator function
def generator(data, batch_size = 32):
    num_samples = len(data)
    while True: 
        shuffle(data)
        for offset in range(0, num_samples, batch_size):
            samples = data[offset: offset + batch_size]

            images = []
            measurements = []
            correction = 0.2
            for sample in samples:
                    for i in range(0,3):          
                        name = './data/IMG/' + sample[i].split('/')[-1]
                        center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) 
                        measurement = float(sample[3])
                        images.append(center_image) 
                        images.append(cv2.flip(center_image, 1)) 

                        # Adding correction factor of 0.2
                        if(i==0):
                            measurements.append(measurement)
                        elif(i==1):
                            measurements.append(measurement + correction) 
                        elif(i==2):
                            measurements.append(measurement - correction)
                        
                        if(i==0):
                            measurements.append(-(measurement))
                        elif(i==1):
                            measurements.append(-(measurement + correction))
                        elif(i==2):
                            measurements.append(-(measurement - correction)) 

            X_train = np.array(images)
            y_train = np.array(measurements)
            
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


# MODEL
model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape = (160,320,3)))

model.add(Lambda(lambda x: x/255.0 - 0.5))


model.add(Convolution2D(24,5,5,subsample=(2,2),activation='elu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='elu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='elu'))
model.add(Convolution2D(64,3,3,activation='elu'))
model.add(Convolution2D(64,3,3,activation='elu'))
model.add(Flatten())
model.add(Dense(100,activation='elu'))
model.add(Dropout(0.25))
model.add(Dense(50,activation='elu'))
model.add(Dense(10,activation='elu'))
model.add(Dense(1))

#adam = optimizers.Adam(lr=0.001)
model.compile(optimizer='adam', loss='mse')
#model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
history_object = model.fit_generator(train_generator,
                    steps_per_epoch=np.ceil(len(train_samples)/batch_size),
                    validation_data=validation_generator,
                    validation_steps=np.ceil(len(validation_samples)/batch_size),
                    epochs=3)

print(history_object.history.keys())


# Save model
model.save('model.h5')

# Summary display
model.summary()
