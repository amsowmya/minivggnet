# save the matplotlib backend so figures can be saved in the backgroud
import matplotlib
matplotlib.use("agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras import backend as K
from pyimagesearch.nn.conv import MiniVggNet
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default="output/minivgg_graph.png", help="path to save the graph")
ap.add_argument("-m", "--model", type=str, default="output/minivgg.hdf5", help="path to save the model")
args = vars(ap.parse_args())

((trainX, trainY), (testX, testY)) = cifar10.load_data()

# if K.image_data_format() == "channels_first":
#     trainX = trainX.reshape(trainX[0], 3, 32, 32)
#     testX = testX.reshape(testX[0], 3, 32, 32)
# else:
#     trainX = trainX.reshape(trainX[0], 32, 32, 3)
#     testX = testX.reshape(testX[0], 32, 32, 3)

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
              "horse", "ship", "truck"]

opt = SGD(learning_rate=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVggNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=40, batch_size=64)

model.save(args["model"])

preds = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=labelNames))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training loss and accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("loss/accuracy")
plt.legend()
plt.savefig(args["output"])

