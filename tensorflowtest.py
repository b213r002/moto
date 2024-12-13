import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# モデルの構築
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

# モデルのコンパイル
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# モデルの要約を表示
model.summary()
