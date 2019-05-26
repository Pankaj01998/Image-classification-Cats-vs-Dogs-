from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
	shear_range = 0.2,
	zoom_range = 0.2,
	horizontal_flip = True
	)
print train_datagen

# test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
	target_size = (64, 64),
	batch_size = 32,
	class_mode = 'binary',
	save_to_dir = 'a/img'
	)

print training_set

i = 0
for batch in train_datagen.flow_from_directory('dataset/training_set',
	target_size = (64, 64),
	batch_size = 32,
	class_mode = 'binary',
	save_to_dir = 'a/img'
	):

    i += 1
  
# test_set = test_datagen.flow_from_directory('dataset/test_set',
# 	target_size = (64, 64),
# 	batch_size = 32,
# 	class_mode = 'binary'
# 	)




# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense

# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(64,64,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

# #this model so far outputs #D feature maps (height, width, features)

# model.add(Flatten()) #this converts our 3D feature maps to 1D
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# from keras.callbacks import ModelCheckpoint
# #storing checkpoints after each epoch of model
# filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

# model.summary()

# model.fit_generator(training_set,
# 	steps_per_epoch = 8000,
# 	epochs = 25,
# 	validation_data = test_set,
# 	validation_steps = 2000, 
# 	callbacks=callbacks_list
# 	)
