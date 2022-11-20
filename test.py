from pyfacerecall import FaceRecognition


if __name__ == '__main__':
    model_path = "model"
    image_path = 'test.jpeg'
    face_recognition = FaceRecognition("./dataset/training", "./dataset/testing")
