import tensorflow as tf

def U_NET(input_shape=(512, 512, 3), classes=3,batch_size=4):
    inputs = tf.keras.Input(input_shape)

    db0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(inputs)
    dc0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(db0)

    da1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2) ,strides=2)(dc0)
    db1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(da1)
    dc1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(db1)

    da2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2) ,strides=2)(dc1)
    db2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(da2)
    dc2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(db2)

    da3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2) ,strides=2)(dc2)
    db3 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(da3)
    dc3 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(db3)
    dc3 = tf.keras.layers.Dropout(0.5)(dc3)

    da4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2) ,strides=2)(dc3)
    db4 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(da4)
    dc4 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(db4)
    dc4 = tf.keras.layers.Dropout(0.5)(dc4)

    ua3 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(2, 2), kernel_initializer='he_normal', activation='relu', padding='same', strides=2)(dc4)
    ub3 = tf.keras.layers.concatenate([ua3, dc3])
    uc3 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(ub3)
    ud3 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(uc3)

    ua2 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(2, 2), kernel_initializer='he_normal', activation='relu', padding='same', strides=2)(ud3)
    ub2 = tf.keras.layers.concatenate([ua2, dc2])
    uc2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(ub2)
    ud2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(uc2)

    ua1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(2, 2), kernel_initializer='he_normal', activation='relu', padding='same', strides=2)(ud2)
    ub1 = tf.keras.layers.concatenate([ua1, dc1])
    uc1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(ub1)
    ud1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(uc1)

    ua0 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), kernel_initializer='he_normal', activation='relu', padding='same', strides=2)(ud1)
    ub0 = tf.keras.layers.concatenate([ua0, dc0])
    uc0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(ub0)
    ud0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(uc0)

    outputs = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1))(ud0)
    model = tf.keras.Model(inputs, outputs)
    return model
