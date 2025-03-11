import cv2
import matplotlib.pyplot as plt

img = cv2.imread('peacock.jpg')

hist = cv2.calcHist([img], [0], None, [256], [0, 256])
equalized = cv2.equalizeHist(img)
equalized_hist = cv2.calcHist([equalized], [0], None, [256], [0, 256])

cv2.imshow('Original', img)
cv2.imshow('Equalized', equalized)

plt. figure()
plt.plot(hist, color='black')
plt.title('Histogram')

plt.figure()
plt.plot(equalized_hist, color='black')
plt.title('Equalized Histogram')

plt. show()

cv2.waitKey(0)
cv2.destroyAllWindows ()
