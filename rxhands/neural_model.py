import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def custom_euclidean_loss(y_true, y_pred):
    proposed_batch = 8
    return tf.nn.l2_loss(y_true-y_pred) / proposed_batch
    
def upsampling_stage(prev_input, skip_input, n_filters=32, is_output=False):
    
    up = Conv2DTranspose(
                 n_filters,
                 3,
                 strides=2,
                 padding='same')(prev_input)
    
    merge = concatenate([up, skip_input], axis=3)
    
    conv = Conv2D(n_filters,
                  3, 
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(merge)
    conv = BatchNormalization()(conv)
        
    conv = Conv2D(n_filters,
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)

    conv = BatchNormalization()(conv)
    
    if not is_output:
        conv = Conv2D(n_filters,
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)
        conv = BatchNormalization()(conv)
    
    return conv

def downsampling_stage(inputs=None, n_filters=32, dropout_probability=0, max_pooling=True):

    conv = Conv2D(n_filters,
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    
    conv = BatchNormalization()(conv)
    
    conv = Conv2D(n_filters,
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    
    if dropout_probability > 0:
        conv = Dropout(dropout_probability)(conv)
         
        
    if max_pooling:
        next_layer = MaxPooling2D((2,2))(conv)
        
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection


def get_heatmap_model(input_size, n_filters=64, n_classes=2, savefile="model-heatmap-rxhands-raw.h5"):
    ''' Implements a UNET network '''
    assert n_classes > 1
    inputs = Input(input_size)
    
    # ENCODING
    downstage1 = downsampling_stage(inputs, n_filters)
    downstage2 = downsampling_stage(downstage1[0], n_filters*2)
    downstage3 = downsampling_stage(downstage2[0], n_filters*4)
    downstage4 = downsampling_stage(downstage3[0], n_filters*8, dropout_probability=0.3)
    downstage5 = downsampling_stage(downstage4[0], n_filters*16, dropout_probability=0.3, max_pooling=False)
    
    # DECODING
    upstage6 = upsampling_stage(downstage5[0], downstage4[1],  n_filters*8, False)
    upstage7 = upsampling_stage(upstage6, downstage3[1],  n_filters * 4, False)
    upstage8 = upsampling_stage(upstage7, downstage2[1],  n_filters * 2, False)
    upstage9 = upsampling_stage(upstage8, downstage1[1],  n_filters, True)

    outputs = Conv2D(n_classes, kernel_size=1, activation='linear', padding='same')(upstage9)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=custom_euclidean_loss,
                  metrics="mse") 

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        ModelCheckpoint(savefile, verbose=1, 
                        save_best_only=True, save_weights_only=True)
    ]

    return model, callbacks

def load_multi_mask_model(weights_path, input_size, n_classes):
    model, _ = get_heatmap_model(input_size, n_classes=n_classes,
                                 savefile="model-heatmap-rxhands-raw.h5")
    model.load_weights(weights_path)
    return model

def main():
    unet, _ = get_multi_mask_model((256, 256, 1), n_classes=19)
    unet.summary()

if __name__ == "__main__":
    main()
