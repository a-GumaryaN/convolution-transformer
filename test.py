from PIL import Image
import numpy as np
from convolution_transform import *
import time
from convolution_transform_functional_p import *

image = np.asarray(Image.open('image1.jpg'))

col=[]
for i in range(image.shape[0]):
    row=[]
    for j in range(image.shape[1]):
        row.append(image[i,j][0])
    col.append(row)

extracted_image=np.array(col)

edge_detection=np.array([
    [-1,  -1,  -1],
    [-1,  8,  -1],
    [-1,  -1,  -1],
])

sharpness=np.array([
    [0,  -1,  0],
    [-1,  5,  -1],
    [0,  -1,  0],
])

box_blur=np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1],
])

gaussian_blur=np.array([
    [1,2,1],
    [2,4,2],
    [1,2,1],
])

kernel_list={
    "edge_detection":edge_detection,
    "sharpness":sharpness,
    "box_blur":box_blur,
    "gaussian_blur":gaussian_blur,
}

kernel=kernel_list["edge_detection"]

padded_matrix=padding(matrix=extracted_image,pad_size=kernel.shape[0],pad_value=0)
before=time.time()
out_image=convolution_transform(matrix=padded_matrix,kernel=kernel,parameter=1,pad_value=kernel.shape[0])
after=time.time()

im = Image.fromarray(np.uint8(out_image))
im.save("output-image-f-ver-result.jpg")

print("execution time for functional version :")
print(str(after-before) + " s")

ct=Convolution_Transform(image=extracted_image,kernel=kernel,parameter=1)
before=time.time()
ct.apply()
after=time.time()
out_image=ct.convolved_matrix

print("execution time for OOP version :")
print(str(after-before) + " s")

im = Image.fromarray(np.uint8(out_image))
im.save("output-image-OOP-ver-result.jpg")
