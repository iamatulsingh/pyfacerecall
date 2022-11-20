from tensorflow.keras import models
from tensorflow.keras.layers import ZeroPadding2D, MaxPooling2D, Conv2D, Flatten, Dense, Dropout


def get_model(input_size=224, output_size=2):
    model = models.Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(input_size,input_size, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    # model.add(Conv2D(4096, (7, 7), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Conv2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Dense(output_size, activation='softmax'))
    return model


if __name__ == '__main__':
    face_recognition_model = get_model()
    face_recognition_model.build((None, 224, 224, 3))
    print(face_recognition_model.summary())
