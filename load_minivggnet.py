from keras.models import load_model
from keras.utils import img_to_array
import cv2
import argparse
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="animals/dog1.jpg", help="path to image")
ap.add_argument("-m", "--model", type=str, default="output/minivgg.hdf5", help="path to the saved model")
args = vars(ap.parse_args())

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
              "horse", "ship", "truck"]

# imagePath = list(paths.list_images(args["input"]))

img = cv2.imread(args["input"])
image = img.copy()
print(image.shape)
image = cv2.resize(image, (32, 32))
print(image.shape)
image = image.reshape(1, 32, 32, 3)
print(image.shape)
image = image.astype("float") / 255.0

model = load_model(args["model"])
pred = model.predict(image)
idx = pred.argmax(axis=1)

img = cv2.putText(img, f"Label: {labelNames[int(idx)]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("image", img)
cv2.waitKey(0)
