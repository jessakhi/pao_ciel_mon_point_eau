import datetime
import numpy as np
import os
import glob
import rasterio
from rasterio.windows import Window
import cv2
from skimage.transform import resize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
import gc  # Garbage Collector
import matplotlib.pyplot as plt

def ensure_directory_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def scale_min_max(array, min_val=0, max_val=10000):
    return np.clip((array.astype(np.float32) - min_val) / (max_val - min_val), 0, 1)

def split_and_scale_images(image_path, prefix, selected_indexes, output_folder, size=256, scale=10000, is_s1=False):
    ensure_directory_exists(output_folder)
    with rasterio.open(image_path) as src:
        for (i, j) in selected_indexes:
            window = Window(i, j, min(size, src.width - i), min(size, src.height - j))
            thumbnail = src.read(window=window).astype(np.float32) / scale
            output_path = f"{output_folder}/{prefix}_tile_{i}_{j}.tif"
            with rasterio.open(output_path, 'w', driver="GTiff", height=size, width=size, count=src.count, dtype=thumbnail.dtype) as dst:
                dst.write(thumbnail)

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    return Activation('relu')(x)

def white_percentage(img):
    h,w = img.shape[:2]
    val, counts = np.unique(img, return_counts=True)
    if 0 in val:
        return counts[0] / (h*w)
    return 0

def unet_model(input_shape=(256, 256 , 6)):
    inputs = Input(input_shape)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = BatchNormalization() (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    # c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    # c4 = BatchNormalization() (c4)
    # p4 = MaxPooling2D((2, 2)) (c4)

    # c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    # c5 = BatchNormalization() (c5)
    # p5 = MaxPooling2D((2, 2)) (c5)

    # c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p5)
    # c6 = BatchNormalization() (c6)
    # p6 = MaxPooling2D((2, 2)) (c6)

    # mid = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p6)
    mid = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)

    # u7 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (mid)
    # c7 = concatenate([u7, c6])
    # c7 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

    # u8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c7)
    # c8 = concatenate([u8, c5])
    # c8 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

    # u9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c8)
    # c9 = concatenate([u9, c4], axis=3)
    # u9 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

    # u10 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (u9)
    u10 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (mid)
    c10 = concatenate([u10, c3], axis=3)
    u10 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c10)

    u11 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (u10)
    c11 = concatenate([u11, c2], axis=3)
    u11 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c11)

    u12 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (u11)
    c12 = concatenate([u12, c1], axis=3)
    u12 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c12)

    outputs = Conv2D(1, (1, 1)) (u12)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='dice', metrics=['accuracy'])
    return model

## fonction pour pourcentage de imagettes noires 
def load_dataset(data_dir, img_size=(256, 256), scale=10000, n_image=None):
    X, Y = [], []
    image_pattern = '*.tif'
    is_s1 = '/s1' in data_dir  

    # print(f"Looking for files in: {data_dir}")
    img_paths = glob.glob(os.path.join(data_dir, image_pattern))
    # print(f"Found {len(img_paths)} image files.")

    for img_path in img_paths:
        # print(f"Processing file: {img_path}")
        x_exists = False
        with rasterio.open(img_path) as src:
            # Determine the bands to read based on Sentinel-1 or Sentinel-2 data
            bands = [1, 2] if is_s1 else list(range(1, src.count + 1))
            img = src.read(bands)
            img = np.moveaxis(img, 0, -1) 

            # Apply scale and normalization
            img = img.astype(np.float32) / scale
            img = scale_min_max(img)  # Normalize to [0, 1]
            img = resize(img, img_size + (len(bands),), preserve_range=True)
            x_exists = True

        if x_exists:
            mask_path = img_path.replace('/s1/', '/masks/').replace('/s2/', '/masks/').replace('s1', 'mask').replace('s2', 'mask')
            if os.path.exists(mask_path):
                # with rasterio.open(mask_path) as src:c
                #     mask = src.read(1)
                try:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    # plt.imshow(mask, cmap='gray')
                    # plt.show()
                    rate = white_percentage(mask)
                    if rate>0.1 and rate<0.9:
                        mask = resize(mask, img_size, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                        X.append(img)
                        Y.append(mask[..., None])  
                except Exception as e:
                    print(e)
            else:
                print(f"Mask file not found for: {img_path}")

    if n_image is None:
        return np.array(X), np.array(Y)
    return np.array(X[:n_image]), np.array(Y[:n_image])


# Simplification de l'utilisation de la mémoire et amélioration de l'efficacité
gc.collect()
import time

model = unet_model()
model.summary()

data_dir = "/home/r/pao_ciel_mon_point_deau/data_folder/s2"
X, Y = load_dataset(data_dir, n_image=200)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
start_time = time.time()
history = model.fit(X_train, Y_train, batch_size=16, epochs=100, validation_split=0.3)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'{elapsed_time:.01f}s')

save_folder = f"runs/model{datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S')}"
os.mkdir(save_folder)
pred_folder = os.path.join(save_folder, 'predicted_masks')
os.mkdir(pred_folder)
mask_folder = os.path.join(save_folder, 'masks')
os.mkdir(mask_folder)
# image_folder = os.path.join(save_folder, 'image')
# os.mkdir(image_folder)
Y_pred = model.predict(X_test)
# score = accuracy_score(Y_test, Y_pred)
for i in range(X_test.shape[0]):
    # cv2.imwrite(os.path.join(image_folder, f'img_{i+1}.tif'), X_test[i])
    cv2.imwrite(os.path.join(mask_folder, f'mask_{i+1}.tif'), Y_test[i])
    cv2.imwrite(os.path.join(pred_folder, f'pred_{i+1}.tif'), Y_pred[i])

model.save(os.path.join(save_folder,'model.keras'))


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()