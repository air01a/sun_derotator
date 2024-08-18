from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import processing
# Ouvrir l'image BMP avec Pillow
image = Image.open("C:/Users/erwan/Sun/170824/AS_P100/test/test.bmp")
#image_array = tiff.imread("C:/Users/erwan/Sun/170824/AS_P100/2024-08-17-1000_0-U-RGB-Sun_lapl3_ap16802.tif")


#if image.mode != 'RGB':
 #   print("RGB")
 #   image = image.convert('RGB')
# Convertir l'image en tableau NumPy
image_array = np.array(image)
image_array=processing.AstroImageProcessing.wavelet_sharpen(image_array, [100,100,100,1,1],[1,1,1,1,1])
# Afficher le tableau NumPy sous forme d'image
plt.imshow(image_array)
plt.axis('off')  # Optionnel : cacher les axes
plt.show()
