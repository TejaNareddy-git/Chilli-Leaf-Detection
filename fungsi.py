import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU

def make_model(n_classes):
    model = Sequential()
    
    # Lapisan convolutional pertama
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Lapisan convolutional kedua
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Lapisan convolutional ketiga
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Lapisan convolutional keempat
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Lapisan fully connected
    model.add(Flatten())
    
    # Dropout untuk mengurangi overfitting
    model.add(Dropout(0.5))
    
    # Lapisan dense dengan 128 unit
    model.add(Dense(128, activation='relu'))
    
    # Lapisan output sesuai jumlah kelas
    model.add(Dense(n_classes, activation='softmax'))

    return model
