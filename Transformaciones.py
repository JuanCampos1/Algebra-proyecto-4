import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform


def rotate_image(image, degree):
    theta = -np.radians(degree)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                [np.sin(theta), np.cos(theta)]])
    center = 0.5*np.array(image.shape)
    offset = center - np.dot(rotation_matrix, center)
    transformed_image = affine_transform(image, rotation_matrix, offset=offset)

    return transformed_image

def scale_image(image, factor):
    scaling_matrix = np.array([[factor, 0], 
                               [0, factor]])
    center = 0.5*np.array(image.shape)
    offset = center - factor * center
    scaled_image = affine_transform(image, scaling_matrix, offset=offset)

    return scaled_image

def deform_image(image, deformation_matrix):
    center = 0.5*np.array(image.shape)
    offset = center - np.dot(deformation_matrix, center)
    deformed_image = affine_transform(image, deformation_matrix, offset=offset)

    return deformed_image


image = plt.imread("Imagen.jpg")

gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

# Rotar la imagen 45 y 90 grados
rotated_image_45 = rotate_image(gray_image, 45)
rotated_image_90 = rotate_image(gray_image, 90)

# Escalar la imagen a la mitad y a la cuarta parte
scaled_image_05 = scale_image(gray_image, 0.5)
scaled_image_025 = scale_image(gray_image, 0.25)

# Deformar la imagen con dos matrices de deformación diferentes
deformation_matrix1 = np.array([[1, 0.5], [0.5, 1]])
deformation_matrix2 = np.array([[1, 0.2], [0.3, 1]])
deformed_image1 = deform_image(gray_image, deformation_matrix1)
deformed_image2 = deform_image(gray_image, deformation_matrix2)

# Mostar las imágenes
plt.figure(figsize=(18,12))

plt.subplot(2, 3, 1)
plt.imshow(rotated_image_45, cmap='gray')
plt.title('Rotated Image (45 degrees)')

plt.subplot(2, 3, 2)
plt.imshow(rotated_image_90, cmap='gray')
plt.title('Rotated Image (90 degrees)')

plt.subplot(2, 3, 3)
plt.imshow(scaled_image_05, cmap='gray')
plt.title('Scaled Image (factor 0.5)')

plt.subplot(2, 3, 4)
plt.imshow(scaled_image_025, cmap='gray')
plt.title('Scaled Image (factor 0.25)')

plt.subplot(2, 3, 5)
plt.imshow(deformed_image1, cmap='gray')
plt.title('Deformed Image (matrix1)')

plt.subplot(2, 3, 6)
plt.imshow(deformed_image2, cmap='gray')
plt.title('Deformed Image (matrix2)')

#La rotación cambia la orientación de la imagen.
#La escala cambia el tamaño de la imagen.
#La deformación cambia la forma de la imagen, estirándola o comprimiéndola en diferentes direcciones.
plt.show()