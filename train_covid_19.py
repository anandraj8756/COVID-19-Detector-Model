import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
  

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="covid19.model",
	help="path to output COVID-19 model")

args = vars(ap.parse_args())

INIT_LR = 1e-3 # learning rate
EPOCHS = 10 # no of eopches to train
BS = 8 # batch size 

print("LOADING IMAGES.......")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels =[]

#loop for the image paths
for imagePath in imagePaths:
	#extract the class label from the file
	label = imagePath.split(os.path.sep)[-2]

	#load the image
	#swap color channels and resize it
	#fixed 224*224 pixels while ingorning aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	#update the data and labes lists
	data.append(image)
	labels.append(label)

#convert the data and labels to numpay
#while scalling the pixel to the range [0, 1]
data = np.array(data) / 255.0
labels = np.array(labels)

# performs one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#partition the data for tranning(80%)
#and testing(20%) using splits
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

#initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=15, fill_mode="nearest")

#load VGG16 network
baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

#constructing the head of the model that will be placed
#on the top of the model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#place the head FC model on top of the base model
#it will become the actual model will be train
model = Model(inputs=baseModel.input, outputs=headModel)

#loop over all layers in the base model
for layer in baseModel.layers:
	layer.trainable = False

#compile our model
print("compileing model.....")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


#train the head of the network
print("tranning head.........")
H = model.fit_generator(
	trainAug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data = (testX, testY),
	validation_steps = len(testX) // BS,
	epochs = EPOCHS)


# make predications on the testing set
print("evaluating network.....")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need 
# to find the the index of the label with 
# corresponding largest predicated probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# compute the confusion matrix nad use it 
# to drive the raw accuracy, sensitivity 
# and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1] / total)
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 0])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the cnfusion matrix, accuracy
#sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# plot the tranning loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")	
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")	
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")	
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Tranning Loss and Accuracy of COVID_19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])	

# serialize the model to disk
print("saving COVID-19 detector model....")
model.save(args["model"], save_format="h5")

















