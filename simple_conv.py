"""
By Abhisek Jana
code taken from https://github.com/adeveloperdiary/blog/tree/master/Computer_Vision/Sobel_Edge_Detection
blog http://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/

Modified by Benjamin Valdes
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
 
def conv_helper(fragment, kernel):
    """ multiplica 2 matices y devuelve su suma"""
    
    f_row, f_col, yes = fragment.shape
    k_row, k_col = kernel.shape 
    result = 0.0
    for row in range(f_row):
        for col in range(f_col):
            result += fragment[row,col] *  kernel[row,col]
    return result

def convolution(image, kernel):
    """Aplica una convolucion sin padding (valida) de una dimesion 
    y devuelve la matriz resultante de la operación
    """

    image_row, image_col, yes = image.shape #asigna alto y ancho de la imagen 
    kernel_row, kernel_col = kernel.shape #asigna alto y ancho del filtro
   
    output = np.zeros(image.shape) #matriz donde guardo el resultado
   
    for row in range(image_row):
        for col in range(image_col):
                output[row, col] = conv_helper(
                                    image[row:row + kernel_row, 
                                    col:col + kernel_col],kernel)
             
    plt.imshow(output, cmap='gray')
    plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
    plt.show()
 
    return output

kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
print(kernel.shape)
Turquia = cv2.imread("C:/Users/windw/Documents/proyecto_Tec/semena-tec-tools-vision/Scripts/Proyecto Vision/Turquia.JPG")
print(Turquia.shape)
convolution(Turquia,kernel)