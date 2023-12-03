import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd

def compress_image(image, k):
    # Convert the image to grayscale if it is RGB
    if len(image.shape) > 2:
        image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

    # Perform the SVD
    U, S, V = svd(image)

    # Keep the top k singular values
    compressed_image = np.dot(U[:,:k], np.dot(np.diag(S[:k]), V[:k,:]))

    return compressed_image

# Load the image
image = plt.imread("Imagen.jpg")

# Compress the image with different values of k
compressed_image_10 = compress_image(image, 10)
compressed_image_25 = compress_image(image, 25)
compressed_image_50 = compress_image(image, 50)
compressed_image_100 = compress_image(image, 100)

#El valor k representa el número de valores singulares que se mantienen durante la compresión. 
#Al aumentar el valor de k, la imagen comprimida se parece más a la imagen original. 
# Esto se debe a que se mantienen más valores singulares, que capturan más detalles de la imagen original.

#El procedimiento resulta en una compresión de la imagen porque se descartan los valores singulares más pequeños, 
# que contribuyen menos a la imagen original. Esto reduce el tamaño de la imagen, 
# pero también puede resultar en una pérdida de detalles.


# Display the compressed images
plt.figure(figsize=(20,10))

plt.subplot(2, 2, 1)
plt.imshow(compressed_image_10, cmap='gray')
plt.title('Compressed Image (k=10)')

plt.subplot(2, 2, 2)
plt.imshow(compressed_image_25, cmap='gray')
plt.title('Compressed Image (k=200)')

plt.subplot(2, 2, 3)
plt.imshow(compressed_image_50, cmap='gray')
plt.title('Compressed Image (k=50)')

plt.subplot(2, 2, 4)
plt.imshow(compressed_image_100, cmap='gray')
plt.title('Compressed Image (k=100)')

plt.show()