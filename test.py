import os
from pyfacerecall import FaceRecognition


if __name__ == '__main__':
    model_path = os.path.join('model')
    image_path = os.path.join('test.jpeg')
    video_path = os.path.join('test.mp4')
    # for training
    # face_recognition = FaceRecognition('./dataset/training', './dataset/testing')
    # after you have your model
    face_recognition = FaceRecognition(number_of_classes=2)
    print(face_recognition.model_prediction(image_path, model_path, False))
    # face_recognition.realtime_prediction(video_path, model_path)
