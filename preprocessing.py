from sklearn.model_selection import train_test_split
import cv2
import os
import yaml

root_dir = "datasets/car_number_plate"
valid_formats = [".jpg", ".jpeg", ".png", ".txt"]

data = {
    "path": "../datasets",
    "train": "images/train",
    "val": "images/valid",
    "test": "images/test",
    "names": ["number-plate"],
}


def file_paths(root, valid_formats):
    file_paths = []

    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            extension = os.path.splitext(filename)[1].lower()

            if extension in valid_formats:
                file_paths.append(os.path.join(dirpath, filename))

    return file_paths


def write_to_file(images_path, labels_path, X):
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    for img_path in X:
        img_name = img_path.split("/")[-1].split(".")[0]
        img_extension = img_path.split("/")[-1].split(".")[1]

        image = cv2.imread(img_path)

        cv2.imwrite(f"{images_path}/{img_name}.{img_extension}", image)

        f = open(f"{labels_path}/{img_name}.txt", "w")
        label_file = open(f"{root_dir}/labels/{img_name}.txt", "r")
        f.write(label_file.read())
        f.close()
        label_file.close()


image_paths = file_paths(root_dir + "/images", valid_formats)
label_paths = file_paths(root_dir + "/labels", valid_formats[-1])

X_train, X_val_test, y_train, y_val_test = train_test_split(
    image_paths, label_paths, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_val_test, y_val_test, test_size=0.7, random_state=42
)

write_to_file("datasets/images/train", "datasets/labels/train", X_train)
write_to_file("datasets/images/valid", "datasets/labels/valid", X_val)
write_to_file("datasets/images/test", "datasets/labels/test", X_test)

# with open("number-plate.yaml", "w") as f:
#     yaml.dump(data, f)
