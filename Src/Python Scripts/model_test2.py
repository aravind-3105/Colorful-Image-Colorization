import keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

from config import img_rows, img_cols, num_classes, kernel

l2_reg = l2(1e-3)

def build_model():
    input_tensor = Input(shape=(img_rows, img_cols, 1))
    x = input_tensor
    
    for i,filters in enumerate([64, 128, 256, 512, 512, 512, 256, 128]):
        if i in [3,4,5,6]:
            continue
        for _ in range(3 if i>1 else 2):
            strideval = (1,1)
            if(i<2 and _==1) or (i==2 and _==2):
                strideval = (2,2)
            dilationval = 2 if filters == 512 else 1
            x = Conv2D(filters, (kernel, kernel), activation='relu', padding='same',
                       kernel_initializer="he_normal", kernel_regularizer=l2_reg,
                       dilation_rate=dilationval,
                       strides=strideval)(x)
        x = BatchNormalization()(x)
        if i==6:
            x = UpSampling2D(size=(2, 2))(x)
            
    outputs = Conv2D(num_classes, (1, 1), activation='softmax', padding='same', name='pred')(x)

    model = Model(inputs=input_tensor, outputs=outputs, name="ColorNet")
    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        encoder_decoder = build_model()
    print(encoder_decoder.summary())
    plot_model(encoder_decoder, to_file='encoder_decoder.svg', show_layer_names=True, show_shapes=True)

    parallel_model = encoder_decoder
    print(parallel_model.summary())
    plot_model(parallel_model, to_file='parallel_model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
