from pyfacerecall import FaceRecognition


if __name__ == '__main__':
    model_path = "model"
    image_path = 'test.jpeg'
    face_recognition = FaceRecognition("./dataset/training", "./dataset/testing")
    face_recognition.training()
    face_recognition.save_model(model_path)
    model = FaceRecognition.load_saved_model(model_path)
    k, result = FaceRecognition.model_prediction(image_path, model_path)
    print(f"detected class is {k} and prediction percentage is {result}")
