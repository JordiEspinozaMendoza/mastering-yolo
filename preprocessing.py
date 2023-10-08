from sklearn.model_selection import train_test_split
import cv2
import os
import yaml

root_dir = "datasets/car_number_plate/"
valid_formats = [".jpg", ".jpeg", ".png", ".txt"]


def file_paths(root, valid_formats):
    file_paths = []

    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            extension = os.path.splitext(filename)[1].lower()

            if extension in valid_formats:
                file_paths.append(os.path.join(dirpath, filename))

    return file_paths
