from os.path import join, exists
from os import makedirs, listdir
import glob
import mtcnn
import argparse

from PIL import Image
import numpy as np
import cv2


def save_cropped_face(images_root_folder, cropped_folder,
                      required_size=(224, 224)):

    if not exists(images_root_folder):
        return Exception("Input Images folder is not exist.")
    file_types = ["*.png", "*.PNG", "*.JPEG", "*.jpeg", "*.jpg", "*.JPG"]
    people = listdir(images_root_folder)

    for file_type in file_types:
        for person in people:
            for i, image_file in enumerate(glob.glob( \
                    join(images_root_folder, person, file_type) \
                    ) \
                    ):

                print(f"processing {image_file}")
                try:
                  img = cv2.imread(image_file)
                  detector = mtcnn.MTCNN()
                  results = detector.detect_faces(img)
                  if not results:
                      continue

                  x, y, width, height = results[0]['box']
                  face = img[y:y + height, x:x + width]
                  try:
                      image = Image.fromarray(face)
                  except ValueError:
                      continue
                  image = image.resize(required_size)
                  face_array = np.asarray(image)

                  makedirs(join(cropped_folder, person), exist_ok=True)

                  output_file_name = f"{person}_{i}{image_file[-4:]}"
                  cv2.imwrite(
                      join(cropped_folder, person, output_file_name),
                      face_array)
                except mtcnn.exceptions.invalid_image.InvalidImage:
                  print("Read error")


def get_detected_face(filename, required_size=(224, 224)):
    img = filename
    if isinstance(filename, str):
        img = cv2.imread(filename)
    detector = mtcnn.MTCNN()
    results = detector.detect_faces(img)
    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array, face


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True,
        help="path to output directory")
    ap.add_argument("-d", "--dataset", default="dataset/training", required=False,
        help="path to save cropped dace dataset")
    args = vars(ap.parse_args())
    save_cropped_face(args["output"], cropped_folder=args["dataset"])
