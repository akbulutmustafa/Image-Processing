import numpy as np
import matplotlib.pyplot as plt
import os


def get_jpeg_files():
    os.getcwd()
    os.listdir()
    path = os.getcwd()
    jpg_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
    return jpg_files
get_jpeg_files()

def compared_ndarray():
    list1=[1,"2hasgdaskjdh,3",'4',5,6]
    list2=[2,"2ashda",3,'114',115,116]
    print(list1+list2)

    list1=[1,2,3,4,5,6]
    list2=[1,2,3,4,5,6]
    print(list1+list2+[10])

    list3=np.asarray([1,2,3,4,5])
    list4=np.asarray([1,2,3,4,5])
    print(list3+list4+10)
compared_ndarray()

image_2 = image_1-30
image_1.shape, image_2.shape
image_1[25,200,:],image_2[25,200,:]
plt.figure(figsize=(3,4))
plt.imshow(image_1)
plt.show()
plt.figure(figsize=(3,4))
plt.imshow(image_2)
plt.show()

def disp_two_img(img1, img2):
    plt.subplot(1,2,1)
    plt.imshow(img1)

    plt.subplot(1,2,2)
    plt.imshow(img2)
    
    plt.show()
disp_two_img(image_1, image_2+30)

def rotate(img1):
    m,n,k = img1.shape
    new_image=np.zeros((n,m,k), dtype='uint8')
    for i in range(m):
        for j in range(n):
            new_image[j][i]=img1[i][j]
    return new_image
    
img3 = rotate(image_1)
disp_two_img(image_1, img3)
