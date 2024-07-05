import datetime
import numpy as np
import os
import glob
import rasterio
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
import cv2
from skimage.transform import resize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import gc  # Garbage Collector
import matplotlib.pyplot as plt

def ensure_directory_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def scale_min_max(array, min_val=0, max_val=1):
    return np.clip((array - min_val) / (max_val - min_val), 0, 1)


def conv2d_block(input_tensor, num_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x


def dice_coeff(y_true, y_pred):
    smooth = 1.0  # smooth term to avoid division by zero
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)

def dice_bce_loss(y_true, y_pred):
    dice_loss_value = dice_loss(y_true, y_pred)
    bce_loss_value = binary_crossentropy(y_true, y_pred)
    return 0*dice_loss_value + bce_loss_value

# When compiling your model, specify the custom loss function

def unet_model(input_shape=(256, 256, 8), num_classes=1, dropout=0.5, batchnorm=True):
    inputs = Input(input_shape)

    # Contracting Path
    c1 = conv2d_block(inputs, 16, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, 32, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, 64, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, 128, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2)) (c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, 256, batchnorm=batchnorm)
    p5 = MaxPooling2D((2, 2)) (c5)
    p5 = Dropout(dropout)(p5)

    c6 = conv2d_block(p5, 512, batchnorm=batchnorm)

    # Expansive Path
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c5])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, 256, batchnorm=batchnorm)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c4])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, 128, batchnorm=batchnorm)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c3])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, 64, batchnorm=batchnorm)

    u10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c9)
    u10 = concatenate([u10, c2])
    u10 = Dropout(dropout)(u10)
    c10 = conv2d_block(u10, 32, batchnorm=batchnorm)

    u11 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c10)
    u11 = concatenate([u11, c1])
    u11 = Dropout(dropout)(u11)
    c11 = conv2d_block(u11, 16, batchnorm=batchnorm)

    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid') (c11)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=dice_bce_loss, metrics=['accuracy'])

    return model



def load_dataset(data_dir, img_size=(256, 256), image_debut=0,image_fin=None):
    X, Y = [], []
    s1_pattern = os.path.join(data_dir, 's1', '*.tif')
    s2_pattern = os.path.join(data_dir, 's2', '*.tif')
    mask_pattern = os.path.join(data_dir, 'masks', '*mask*.tif')

    s1_paths = glob.glob(s1_pattern)
    s2_paths = glob.glob(s2_pattern)
    mask_paths = glob.glob(mask_pattern)

    print("S1 paths found:", s1_paths[:3])
    print("S2 paths found:", s2_paths[:3])
    print("Mask paths found:", mask_paths[:3])

    paired_paths = []
    for mask_path in mask_paths:
        base_name = os.path.basename(mask_path).replace('mask', 's1')
        s1_path = next((s for s in s1_paths if os.path.basename(s) == base_name), None)
        base_name = os.path.basename(mask_path).replace('mask', 's2')
        s2_path = next((s for s in s2_paths if os.path.basename(s) == base_name), None)
        
        if s1_path and s2_path:
            paired_paths.append((s1_path, s2_path, mask_path))

    print("Paired paths:", paired_paths[:3])

    for s1_path, s2_path, mask_path in paired_paths:
        with rasterio.open(s1_path) as src_s1, rasterio.open(s2_path) as src_s2, rasterio.open(mask_path) as src_mask:
            s1_image = src_s1.read()
            s2_image = src_s2.read()
            mask = src_mask.read(1)

            s1_image = np.moveaxis(s1_image, 0, -1).astype(np.float32) / 100
            s2_image = np.moveaxis(s2_image, 0, -1).astype(np.float32) / 10000
            s1_image = scale_min_max(s1_image)
            s2_image = scale_min_max(s2_image)
            combined_image = np.concatenate((s1_image, s2_image), axis=-1)

            combined_image = resize(combined_image, img_size + (combined_image.shape[-1],), preserve_range=True)
            mask = (resize(mask, img_size, preserve_range=True) > 0.5).astype(np.uint8)

            X.append(combined_image)
            Y.append(mask[..., np.newaxis])

    X, Y = np.array(X), np.array(Y)
    print("Final datasets shapes:", X.shape, Y.shape)
    if image_fin is not None:
        X, Y = X[image_debut:image_fin], Y[image_debut:image_fin]
    else:
        X, Y = X[image_debut:], Y[image_debut:]
    return X, Y

gc.collect()
import time

model = unet_model(input_shape=(256, 256, 8))  # Update the number of channels as per combined S1 and S2
model.summary()

data_dir = r'data_folder/data_folder'
X, Y = load_dataset(data_dir, img_size=(256, 256), image_debut=0, image_fin=100)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

start_time = time.time()
batchTrain = 100

results = []

#for k in range(6):
#Xk_train = X_train[k*batchTrain:(k+1)*batchTrain]
#Yk_train = Y_train[k*batchTrain:(k+1)*batchTrain]
#history = model.fit(Xk_train, Yk_train, batch_size=16, epochs=15, validation_split=0.3)
history = model.fit(X_train, Y_train, batch_size=16, epochs=5, validation_split=0.3)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'{elapsed_time:.01f}s')

save_folder = f"runs/model{datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S')}"
graph_folder = f"results/graphs{datetime.datetime.now().strftime('%d_%H-%M-%S')}"
ensure_directory_exists(graph_folder)
ensure_directory_exists(save_folder)
pred_folder = os.path.join(save_folder, 'predicted_masks')
ensure_directory_exists(pred_folder)
mask_folder = os.path.join(save_folder, 'masks')
ensure_directory_exists(mask_folder)

#Yk_pred = model.predict(X_test)
Y_pred = model.predict(X_test)
Y_pred_train = model.predict(X_train)

loss_train = []
loss_test = []
loss_train.append((binary_crossentropy(Y_train,Y_pred_train),
                  dice_loss(Y_train,Y_pred_train)))
loss_test.append((binary_crossentropy(Y_train,Y_pred_train),
                 dice_loss(Y_train,Y_pred_train)))

print('loss : bce , dice')
print(f'loss_train : {loss_train}')
print(f'loss_test : {loss_test}')
#thresholded_pred = (Yk_pred > np.median(Yk_pred)).astype(np.uint8)

thresholded_pred = []
thresholded_pred_train = []
for img in Y_pred:
    thresholded_pred.append(cv2.threshold((255*img).round().astype('uint8'),0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
    #thresholded_pred.append((255*img).round().astype('uint8'))
for img in Y_pred_train:
    thresholded_pred_train.append(cv2.threshold((255*img).round().astype('uint8'),0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
    #thresholded_pred_train.append((255*img).round().astype('uint8'))

for i in range(X_test.shape[0]):
    cv2.imwrite(os.path.join(mask_folder, f'mask_{i+1}.tif'), Y_test[i])
    #cv2.imwrite(os.path.join(pred_folder, f'pred_{i+1}.tif'), Yk_pred[i])
    cv2.imwrite(os.path.join(pred_folder, f'pred_{i+1}.tif'), Y_pred[i])

#plt.hist(Y_pred[1].ravel(), bins=256, density=True)
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 15))
# Afficher chaque image dans une des sous-graphes
for i, ax in enumerate(axes.flat):
    ax.hist(Y_train[i].ravel(), bins=256, density=True)
plt.title('Histogramme large des valeurs des pixels')
plt.xlabel('Valeur du pixel')
plt.ylabel('Fréquence')
plt.savefig(os.path.join(graph_folder,'histoLarge.png'))
plt.show()

#plt.hist(Yk_pred[1].ravel(), bins=256, range=(np.median(Yk_pred[1])-0.1,np.median(Yk_pred[1])+0.1), density=True)
plt.hist(Y_pred[1].ravel(), bins=256, range=(np.median(Y_pred[1])-0.1,np.median(Y_pred[1])+0.1), density=True)
plt.title('Histogramme des valeurs des pixels')
plt.xlabel('Valeur du pixel')
plt.ylabel('Fréquence')
plt.savefig(os.path.join(graph_folder,'histogramme.png'))
plt.show()


fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 15))
# Afficher chaque image dans une des sous-graphes
for i, ax in enumerate(axes.flat):
    ax.imshow(Y_train[i], cmap='gray')  # Afficher l'image en niveaux de gris
plt.title('Y_train')
plt.savefig(os.path.join(graph_folder,'Y_train.png'))
plt.show()

fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15,15))
# Afficher chaque image dans une des sous-graphes
for i, ax in enumerate(axes.flat):
    ax.imshow(Y_pred_train[i], cmap='gray')  # Afficher l'image en niveaux de gris
plt.title('Y_pred_train')
plt.savefig(os.path.join(graph_folder,'Y_pred_train.png'))
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15,15))
# Afficher chaque image dans une des sous-graphes
for i, ax in enumerate(axes.flat):
    ax.imshow(Y_pred[i], cmap='gray')  # Afficher l'image en niveaux de gris
    #ax.imshow(Yk_pred[i], cmap='gray')
plt.title('Y_pred')
plt.savefig(os.path.join(graph_folder,'Y_pred.png'))
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15,15))
# Afficher chaque image dans une des sous-graphes
for i, ax in enumerate(axes.flat):
    ax.imshow(thresholded_pred[i], cmap='gray')  # Afficher l'image en niveaux de gris
plt.title('thresholded_pred')
plt.savefig(os.path.join(graph_folder,'thresholded_pred.png'))
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15,15))
# Afficher chaque image dans une des sous-graphes
for i, ax in enumerate(axes.flat):
    ax.imshow(thresholded_pred_train[i], cmap='gray')  # Afficher l'image en niveaux de gris
plt.title('thresholded_pred_train')
plt.savefig(os.path.join(graph_folder,'thresholded_pred_train.png'))
plt.show()

model.save(os.path.join(save_folder,'model.keras'))
results.append(history.history)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(graph_folder,'accuracy.png'))
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(graph_folder,'loss.png'))
plt.show()