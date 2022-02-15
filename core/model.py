from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def conv_block(x, num_filters, use_bn=True):
    x = Conv2D(num_filters, kernel_size=3, padding="same")(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(num_filters, kernel_size=3, padding="same")(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    return x

def encoder_block(x, num_filters, use_bn=True):
    x = conv_block(x, num_filters, use_bn)
    p = MaxPool2D(pool_size=(2, 2))(x)
    
    return x, p

def decoder_block(x, skip_feature, num_filters,use_bn=True):
    x = Conv2DTranspose(filters=num_filters, kernel_size=(2, 2), strides=2, padding="same")(x)
    x = Concatenate()([x, skip_feature])
    
    x = conv_block(x, num_filters, use_bn)
    return x

def build_unet(input_shape, filter_sizes, classes, use_bn=True):
    inputs = Input(input_shape)
    skip_blocks = []
    total_filters = len(filter_sizes)
    
    for index, f in enumerate(filter_sizes):
        # If it's the first filter, send inputs as x
        if index == 0:
            s, x = encoder_block(inputs, f, use_bn=use_bn)
            skip_blocks.append(s)
        
        elif index != total_filters-1:
            s, x = encoder_block(x, f, use_bn=use_bn)
            skip_blocks.append(s)
            
        else:
            x = conv_block(x, f, use_bn=use_bn)
    
    # Reverse the skip blocks so that we can match the skip connections
    # with the correct decoded block
    skip_blocks = skip_blocks[::-1]
    
    for index, f in enumerate(filter_sizes[::-1]):
        # We don't want the first filter as it was only the base layer
        if index == 0:
            continue
        
        x = decoder_block(x, skip_blocks[index-1], f, use_bn=use_bn)
    
    if classes == 1:
        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(x)
    else:
        outputs = Conv2D(classes, 1, padding="same", activation="softmax")(x)
    
    model = Model(inputs, outputs, name="U-Net")
    
    return model

