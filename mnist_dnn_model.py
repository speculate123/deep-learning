from sklearn.datasets import fetch_openml
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import pytesseract
import pyautogui
import matplotlib.image as mpimg

mnist = fetch_openml('mnist_784')
data = mnist['data']
target = mnist['target']

import numpy as np
data_train = data[:60000]
target_train = target[:60000]
data_test = data[60000:]
target_test_temp = target[60000:]

shuffle = np.random.permutation(60000)
data_train = data_train[shuffle]
target_train_temp = target_train[shuffle]

target_train = []
for i in range(len(target_train_temp)):
    temp = []
    for j in range(10):
        if j == int(target_train_temp[i]):
            temp.append(1)
        else:
            temp.append(0)
    target_train.append(temp)
target_train = np.array(target_train)

target_test = []
for i in range(len(target_test_temp)):
    temp = []
    for j in range(10):
        if j == int(target_test_temp[i]):
            temp.append(1)
        else:
            temp.append(0)
    target_test.append(temp)
target_test = np.array(target_test)

#DNN
from keras import Sequential
from keras.layers import Dense, Activation
model = Sequential()
#加一層fully connected layer(dense) 
#model.add(Dense(input_dim = 28*28, units = 500))
model.add(Dense(500, input_dim = 28*28))
model.add(Activation('sigmoid'))
#再加一層fully connected layer(dense), 不用input, 跟前一層output一樣
model.add(Dense(500))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

#evaluate model好壞
#configuration(optimizer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#batch越大，速度越快，但是結果會爛掉
#batch越小，速度越慢，gpu平行運算加速效果不明顯
model.fit(data_train, target_train, batch_size=100, epochs=20)

score = model.evaluate(data_test, target_test)

im = Image.fromarray(data_test[4].reshape(28,28))
plt.imshow(im)

predict = model.predict(data_test)[4]
np.where(predict == np.amax(predict))

# read handwrite image
data = Image.open(r'C:\Users\yt335\Desktop\handwriting number.png').convert('L')
data_np = np.array(data).reshape(1,784)
data_np = data_np/255
predict = model.predict(data_np)
np.where(predict == np.amax(predict))

#OCR零件文字辨識
imagepath = 'test4.jpg'

img = cv2.imread(imagepath) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = cv2.resize(img, (28, 28))
#blueImg = img[:,:,0]
#b = get_blue(img)
#img = cv2.cvtColor(b, cv2.IMREAD_GRAYSCALE)
ret,img = cv2.threshold(img, 130,255,cv2.THRESH_BINARY_INV)

dst = cv2.fastNlMeansDenoising(img,None,10,7,21)

median = cv2.medianBlur(dst, 5)
kernel_size = 3
img = cv2.GaussianBlur(median,(kernel_size, kernel_size), 0)
#cv2.imwrite('output.jpg', img)
img = img/255

plt.imshow(img)

test = np.array(img).reshape(1,784)
predict = model.predict(test)
np.where(predict == np.amax(predict))


