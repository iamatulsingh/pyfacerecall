import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pyfacerecall.model import get_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

from pyfacerecall.face_detection_operation import get_detected_face


class FaceRecognition:

    def __init__(self, training_path="", testing_path=""):
        self.TRAINING_DATA_DIRECTORY = training_path
        self.TESTING_DATA_DIRECTORY = testing_path
        self.EPOCHS = 50
        self.BATCH_SIZE = 8
        self.NUMBER_OF_TRAINING_IMAGES = 64
        self.NUMBER_OF_TESTING_IMAGES = 24
        self.IMAGE_HEIGHT = 224
        self.IMAGE_WIDTH = self.IMAGE_HEIGHT
        self.OUTPUT_LAYERS = len([name for name in 
          os.listdir(self.TRAINING_DATA_DIRECTORY) if not os.path.isfile(name)])
        self.model = get_model(self.IMAGE_HEIGHT, self.OUTPUT_LAYERS)
        self.training_generator = None

    @staticmethod
    def plot_training(history):
        plot_folder = "plot"
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.1, 1])
        plt.legend(loc='lower right')

        if not os.path.exists(plot_folder):
            os.mkdir(plot_folder)

        plt.savefig(os.path.join(plot_folder, "model_accuracy.png"))

    @staticmethod
    def data_generator():
        img_data_generator = ImageDataGenerator(
            rescale=1./255,
            # horizontal_flip=True,
            fill_mode="nearest",
            # zoom_range=0.3,
            # width_shift_range=0.3,
            # height_shift_range=0.3,
            rotation_range=30
        )
        return img_data_generator

    def training(self):
        self.training_generator = FaceRecognition.data_generator().flow_from_directory(
            self.TRAINING_DATA_DIRECTORY,
            target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical'
        )

        testing_generator = FaceRecognition.data_generator().flow_from_directory(
            self.TRAINING_DATA_DIRECTORY,
            target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            class_mode='categorical'
        )

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9, decay=1e-2 / self.EPOCHS),
            metrics=["accuracy"]
        )

        history = self.model.fit(
            self.training_generator,
            steps_per_epoch=self.NUMBER_OF_TRAINING_IMAGES//self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_data=testing_generator,
            shuffle=True,
            validation_steps=self.NUMBER_OF_TESTING_IMAGES//self.BATCH_SIZE
        )

        FaceRecognition.plot_training(history)

    def save_model(self, model_path):
        self.model.save(model_path)
        class_names = self.training_generator.class_indices
        class_names_file_reverse = "class_names_reverse.npy"
        class_names_file = "class_names.npy"
        np.save(os.path.join(model_path, class_names_file_reverse), class_names)
        class_names_reversed = np.load(os.path.join(model_path, class_names_file_reverse), allow_pickle=True).item()
        class_names = dict([(value, key) for key, value in class_names_reversed.items()])
        np.save(os.path.join(model_path, class_names_file), class_names)


    def load_saved_model(self, model_path):
        model = load_model(model_path)
        return model


    def model_prediction(self, image_path, model_path, need_cropping=True):
        class_name = "None Class Name"
        if not need_cropping:
            face = cv2.imread(image_path)
            image = Image.fromarray(face)
            image = image.resize((self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
            face_array = np.asarray(image)
        else:
            face_array, face = get_detected_face(image_path)
        model = load_model(model_path)
        face_array = face_array.astype('float32')
        input_sample = np.expand_dims(face_array, axis=0)
        result = model.predict(input_sample)
        best_result = np.argmax(result, axis=1)
        index = best_result[0]

        classes = np.load(os.path.join(model_path, "class_names.npy"), allow_pickle=True).item()
        # print(classes, type(classes), classes.items())
        if type(classes) is dict:
            for k, v in classes.items():
                if k == index:
                    class_name = v

        return class_name, result


    def realtime_prediction(self, src, model_path, need_cropping=True):
        classes = np.load(os.path.join(model_path, "class_names.npy"), allow_pickle=True).item()
        class_name = "None Class Name"
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print("Error opening video source")
        while cap.isOpened():
            ret, frame = cap.read()
            if not need_cropping:
                image = Image.fromarray(frame)
                image = image.resize((self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
                face_array = np.asarray(image)
            else:
                face_array, face = get_detected_face(face)
            model = load_model(model_path)
            face_array = face_array.astype('float32')
            input_sample = np.expand_dims(face_array, axis=0)
            result = model.predict(input_sample)
            best_result = np.argmax(result, axis=1)
            index = best_result[0]

            # print(classes, type(classes), classes.items())
            if type(classes) is dict:
                for k, v in classes.items():
                    if k == index:
                        class_name = v

            print(class_name, result)
        cap.release()
