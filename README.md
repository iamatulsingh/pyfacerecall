# pyfacerecall

[![ForTheBadge built-with-love](https://forthebadge.com//images/badges/built-with-love.svg)](https://github.com/iamatulsingh/)

## Installation
```bash
pip install pyfacerecall
```

## Generate own face dataset using camera
You can use cli built-in cli tool to generate dataset of your own face using your camera. 
```bash
python -m pyfacerecall --output person --source 0
```

## How to use?
```python
from pyfacerecall import FaceRecognition


if __name__ == '__main__':
    model_path = "model"
    image_path = 'test.jpeg'
    face_recognition = FaceRecognition("./dataset/training", "./dataset/testing")
    face_recognition.training()
    face_recognition.save_model(model_path)
    model = face_recognition.load_saved_model(model_path)
    k, result = face_recognition.model_prediction(image_path, model_path, need_cropping=False)
    print(f"detected class is {k} and prediction percentage is {result}")
```
