# Import necessary libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv

# Read an image and convert it into grayscale
img = mpimg.imread('/content/grasshopper.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Display the original image
plt.imshow(img)
plt.title("Original Image")
plt.show()

# Fetch the dimensions of the image
rows = img.shape[0]
cols = img.shape[1]

# Initialize the Sobel operator kernels
sbx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sby = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# Convolution
outx = []
for i in range(rows - 2):
    for j in range(cols - 2):
        mat = img[(0 + i):(3 + i), (0 + j):(3 + j)] * sbx
        lst = np.sum(mat)
        outx.append(lst)

outy = []
for i in range((rows - 2)):
    for j in range((cols - 2)):
        mat = img[(0 + i):(3 + i), (0 + j):(3 + j)] * sby
        lst = np.sum(mat)
        outy.append(lst)

outx = np.array(outx)
outy = np.array(outy)

# Reshape linear arrays back to their matrix forms of resultant dimensions
outx = np.reshape(outx, ((rows - 2), (cols - 2)))
outy = np.reshape(outy, ((rows - 2), (cols - 2)))

# Calculate the resultant from the x and y gradients (out x and out y respectively)
outf = np.hypot(outx, outy)

# Display the original image
plt.imshow(img, cmap='gray')
plt.title("Original Grayscale Image")
plt.show()

# Display the image portraying the edges
plt.imshow(outf, cmap='gray')
plt.title("Edge-Detected Image")
plt.show()
