import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPool2D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import cv2
import pandas as pd
import random
import os
import ntpath
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.image as npimg
from imgaug import augmenters as iaa

datadir = "Record-track"

columns = ["center", "left","right","steering","throttle","reverse","speed"]
data = pd.read_csv(os.path.join(datadir,"driving_log.csv"),names=columns)
pd.set_option('display.max_columns', 7)
# print(data.head)

def path_leaf(path):
  head,tail = ntpath.split(path)
  return tail

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)

# print(data.head())
# print(data.shape[0])

### Start of Pre Processing

num_bins = 25
samples_per_bin = 250
hist,bins = np.histogram(data['steering'],num_bins)
# print(bins)

centre = (bins[:-1] + bins[1:]) * 0.5
plt.bar(centre,hist,width=0.05)
plt.plot((np.min(data['steering']),np.max(data['steering'])),(samples_per_bin,samples_per_bin))
plt.show()
# print(centre)

remove_list = []
print("Total Data: " , len(data))

for j in range(num_bins):
  list_ = []
  for i in range (len(data['steering'])):
    if bins[j] <= data['steering'][i] <= bins[j+1]:
      list_.append(i)
  list_ = shuffle(list_)
  list_ = list_[samples_per_bin:]
  remove_list.extend(list_)

print("Removed: ", len(remove_list))

### Actually Removing Data

data.drop(data.index[remove_list], inplace=True)
print("Remaining Data: ", len(data))

# hist, _ = np.histogram(data['steering'],num_bins)
# plt.bar(centre,hist,width=0.05)
# plt.plot((np.min(data['steering']),np.max(data['steering'])),(samples_per_bin,samples_per_bin))

# Training and Validation Split
def load_img_steering(datadir, df):
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    # centre,left,right = indexed_data[0], indexed_data[1],indexed_data[2]
    centre, left, right = indexed_data.iloc[0], indexed_data.iloc[1], indexed_data.iloc[2]
    steering.append(float(indexed_data.iloc[3]))
    image_path.append(os.path.join(datadir,centre.strip()))
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings

image_paths, steerings = load_img_steering(datadir+'/IMG',data)
X_train,X_valid,y_train,y_valid = train_test_split(image_paths,steerings,test_size=0.2,random_state=6)

print("Training Samples: ", (len(X_train)),"Validation Samples: ", (len(X_valid)))

fig,axes = plt.subplots(1,2,figsize = (12,4))
axes[0].hist(y_train,bins = num_bins, width=0.05,color='blue')
axes[0].set_title('Training Set')
axes[1].hist(y_valid,bins = num_bins, width=0.05,color='red')
axes[1].set_title('Validation Set')
plt.show()

def zoom(image):
  zoom = iaa.Affine(scale=(1, 1.3))
  image = zoom.augment_image(image)
  return image

image = image_paths[random.randint(0,1000)]
original_image = plt.imread(image)
zoomed_image = zoom(original_image)
fig,axes = plt.subplots(1,2,figsize = (12,4))
axes[0].imshow(original_image)
axes[0].set_title('Original image')
axes[1].imshow(zoomed_image)
axes[1].set_title('zoom image')
plt.show()

def pan(image):
  pan = iaa.Affine(translate_percent={"x":(-0.1,0.1), "y":(-0.1, 0.1)})
  panned_image = pan.augment_image(image)
  return panned_image

image = image_paths[random.randint(0,1000)]
original_image = plt.imread(image)
panned_image = pan(original_image)
fig,axes = plt.subplots(1,2,figsize = (12,4))
axes[0].imshow(original_image)
axes[0].set_title('Original image')
axes[1].imshow(panned_image)
axes[1].set_title('panned image')
plt.show()

def img_random_brightness(image):
  brightness = iaa.Multiply((0.2, 1.2))
  image = brightness.augment_image(image)
  return image

image = image_paths[random.randint(0,1000)]
original_image = plt.imread(image)
bright_image = img_random_brightness(original_image)
fig,axes = plt.subplots(1,2,figsize = (12,4))
axes[0].imshow(original_image)
axes[0].set_title('Original image')
axes[1].imshow(bright_image)
axes[1].set_title('bright image')
plt.show()

def img_random_flip(image, steering_angle):
  image = cv2.flip(image, 1)
  steering_angle = -steering_angle
  return image, steering_angle

random_index = random.randint(0, 1000)
image = image_paths[random_index]
steering_angle = steerings[random_index]
original_image = plt.imread(image)
flipped_image, flipped_steering = img_random_flip(original_image, steering_angle)
fig,axes = plt.subplots(1,2,figsize = (12,4))
axes[0].imshow(original_image)
axes[0].set_title('Original image + sterring angle ' + str(steering_angle))
axes[1].imshow(flipped_image)
axes[1].set_title('flipped image + sterring angle ' + str(flipped_steering))
plt.show()

def random_augment(image, steering_angle):
  image = plt.imread(image)
  if np.random.rand() < 0.5:
    image = zoom(image)
  if np.random.rand() < 0.5:
    image = pan(image)
  if np.random.rand() < 0.5:
    image = img_random_brightness(image)
  if np.random.rand() < 0.5:
    image, steering_angle = img_random_flip(image, steering_angle)
  return image, steering_angle

#Pre Proceess Images
def img_process(img):
  img = npimg.imread(img)
  img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img,(3,3),0)
  img = cv2.resize(img,(200,66))
  img = img/255
  return img

def img_process_no_imread(img):
  img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img,(3,3),0)
  img = cv2.resize(img,(200,66))
  img = img/255
  return img

def batch_generator(image_paths, steering_angles, batch_size, is_training):
  while True:
    batch_img = []
    batch_steering = []
    for i in range(batch_size):
      random_index = random.randint(0, len(image_paths)-1)
      if is_training:
        im, steering = random_augment(image_paths[random_index], steering_angles[random_index])
      else:
        im = plt.imread(image_paths[random_index])
        steering = steering_angles[random_index]
      im = img_process_no_imread(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield(np.asarray(batch_img), np.asarray(batch_steering))



x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))
fig,axes = plt.subplots(1,2,figsize = (12,4))
axes[0].imshow(x_train_gen[0])
axes[0].set_title('Training image')
axes[1].imshow(x_valid_gen[0])
axes[1].set_title('Valid image')
plt.show()

image = image_paths[149]
original_image = npimg.imread(image)
preprocessed_image = img_process(image)

fig,axes = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[1].imshow(preprocessed_image)
axes[1].set_title("Preprocessed Image")
plt.show()

def nvidia_model():
  model = Sequential()
 
  model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
  model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
  model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
  model.add(Convolution2D(64, (3, 3), activation='elu'))
  model.add(Convolution2D(64, (3, 3), activation='elu'))
 
  model.add(Flatten())
  model.add(Dense(100, activation = 'elu'))
  model.add(Dropout(0.5))
  model.add(Dense(50, activation = 'elu'))
  model.add(Dense(10, activation = 'elu'))
  model.add(Dense(1))
 
  model.compile(Adam(lr=0.0001),loss='mse')
  return model

model = nvidia_model()
print("Model Summary")
model.summary()


history = model.fit(batch_generator(X_train, y_train, 100, 1),
                              steps_per_epoch=300,
                              epochs=10,
                              validation_data=batch_generator(X_valid, y_valid, 100, 0),
                              validation_steps=200
                              )



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

model.save('new_model.h5')
