import keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model


l2_reg = l2(1e-3)
img_rows, img_cols =  256, 256
num_classes = 313
kernel = 3

def build_base_model():
    input_tensor = Input(shape=(img_rows, img_cols, 1))
    x = input_tensor
    
    for i,filters in enumerate([64, 128, 256, 512, 512, 512, 256, 128]):
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


def build_ablation_model1():
    input_tensor = Input(shape=(img_rows, img_cols, 1))
    x = input_tensor
    
    for i,filters in enumerate([64, 128, 256, 512, 512, 512, 256, 128]):
        if i in [3,5,6]:
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


def build_ablation_model2():
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


def main():
    with tf.device("/cpu:0"), K.get_session():
        model = build_base_model()

    print(model.summary())

    parallel_model = model
    print(parallel_model.summary())

    K.clear_session()

if __name__ == '__main__':
    main()
