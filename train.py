from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from core.model import build_unet
from core.utils import hex_to_rgb
from core.utils import to_labels
from core.utils import info
from imutils import paths
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import config
import cv2

if config.PATCH_SIZE_X:
    import patchify

info("Loading Image paths")
images_path = list(paths.list_images(config.DATASET_PATH))

images = []
masks = []

if config.PATCH_SIZE_X:
    info("Dividing input images and masks into {}x{} px.".format(config.PATCH_SIZE_X, config.PATCH_SIZE_Y))
    
    for image_path in images_path:
        info("Processing {}".format(image_path))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        x_size = (image.shape[1] // config.PATCH_SIZE_X) * config.PATCH_SIZE_X
        y_size = (image.shape[0] // config.PATCH_SIZE_Y) * config.PATCH_SIZE_Y
        
        image = Image.fromarray(image)
        image = image.crop((0, 0, x_size, y_size))
        image = np.array(image)
        
        patches_image = patchify.patchify(image, (config.PATCH_SIZE_X, config.PATCH_SIZE_Y, 3), step=config.PATCH_SIZE_X)
        
        for i in range(patches_image.shape[0]):
            for j in range(patches_image.shape[1]):
                single_patch = patches_image[i, j, :, :]
                single_patch.reshape(-1, single_patch.shape[-1])
                single_patch = single_patch.astype('float32') / 255.0
                
                # We remove the extra dimesnion patchify adds by only taking the
                # first value.
                if "masks" in image_path:
                    masks.append(single_patch[0])
                else:
                    images.append(single_patch[0])
                    
else:
    for image_path in images_path:
        info("Loading images and masks")
        if "masks" in image_path:
            mask = cv2.imread(image_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = np.array(mask)
            masks.append(mask)
            
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            image = np.array(image)
            images.append(image)
        
images = np.array(images)
masks = np.array(masks)

if config.TEST_IMAGE_PLOT:
    file_index = np.random.randint(0, len(images))
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(images[file_index])
    plt.subplot(122)
    plt.imshow(masks[file_index])
    plt.show()

info("Decoding and loading Labels")
label_metadata = [
    ["building", hex_to_rgb("#3C1098")],
    ["land", hex_to_rgb("#8429F6")],
    ["road", hex_to_rgb("#6EC1E4")],
    ["vegetation", hex_to_rgb("#FEDD3A")],
    ["water", hex_to_rgb("#E2A929")],
    ["unlabeled", hex_to_rgb("#9B9B9B")],
]

labels = []
for i in range(masks.shape[0]):
    label = to_labels(masks[i], label_metadata)
    labels.append(label)
    
labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)
info("Unique labels in label dataset are: {}".format(np.unique(labels)))

if config.TEST_IMAGE_PLOT:
    file_index = np.random.randint(0, len(images))
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(images[file_index])
    plt.subplot(122)
    plt.imshow(labels[file_index])
    plt.show()

info("One hot encoding Labels")
total_classes = len(np.unique(labels))
one_hot_labels = to_categorical(labels, num_classes=total_classes)

info("Splitting the dataset into train and test")
x_train, x_test, y_train, y_test = train_test_split(images, one_hot_labels, test_size=config.TEST_SPLIT)

info("Loading and Compiling Model")
input_shape = (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT, config.NUM_CHANNELS)
model = build_unet(
            input_shape=input_shape,
            filter_sizes=config.FILTERS,
            classes=config.NUM_CLASSES,
            use_bn=config.USE_BN
        )
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

if config.MODEL_SUMMARY:
    model.summary()

H = model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              batch_size=config.BATCH_SIZE,
              epochs=config.NUM_EPOCHS,
              verbose=1
          )

info("Saving Model")
model.save(config.MODEL_PATH)