from keras.applications import resnet50
from keras.preprocessing import image
from keras.models import Model
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Input, Dense, GlobalAveragePooling2D
#from keras.optimizers import Adam
from keras import optimizers
import numpy as np


batch_size = 50
num_classes = 2
img_size = 224

base_model = resnet50.ResNet50

base_model = base_model(weights = 'imagenet', include_top = False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation= 'relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs = base_model.input, outputs = predictions)
'''
for layer in base_model.layers:
    layer.trainable = False
'''
opt = optimizers.SGD(lr=0.001, clipvalue=0.5, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss = 'sparse_categorical_crossentropy', 
            optimizer = opt, 
            metrics = ['acc'])


x_train = np.random.normal(loc = 127, scale = 127, size = (50, img_size, img_size, 3))
y_train = np.array([0,1]*25)
#import bpython
#bpython.embed(locals())
x_train = resnet50.preprocess_input(x_train)


print(model.evaluate(x_train, y_train, batch_size = batch_size, verbose = 0))
model.fit(x_train, y_train,
        epochs = 100,
        batch_size = batch_size,
        shuffle = False,
        validation_data = (x_train, y_train))
