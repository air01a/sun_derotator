from cv2 import GaussianBlur, cvtColor, COLOR_BGR2HSV, COLOR_HSV2BGR, BORDER_DEFAULT, \
    resize
from numpy import zeros, empty,  float32, uint16, percentile, clip, uint8,interp, log, fft, mean, std, squeeze, correlate, sqrt, inf,ones
import cv2
from math import exp
I16_BITS_MAX_VALUE = 65535
from scipy.signal import correlate2d
import imutils
from skimage import measure
import matplotlib.pyplot as plt
from skimage import io, measure, color
from skimage.draw import polygon

class AstroImageProcessing:
    @staticmethod
    def levels(image, blacks: float, midtones: float, whites: float, contrast: float =1, r:float =1, g:float=1, b:float=1):
        min = image.min()
        max = image.max()
        median = max - min
        if midtones <= 0:
            midtones = 0.1
        # midtones
        image = I16_BITS_MAX_VALUE *((((image-min)/median - 0.5) * contrast + 0.5)  * median + min ) ** (1 / midtones) / I16_BITS_MAX_VALUE ** (1 / midtones)
        #black / white levels
        image = clip(image, blacks, whites)

        if (len(image.shape)<3):
            image = float32(interp(image,
                                                (min, max),
                                                (0, I16_BITS_MAX_VALUE)))
        else:
            image[:,:,0] = float32(interp(image[:,:,0],
                                                (min, max),
                                                (0, I16_BITS_MAX_VALUE)))

        if (len(image.shape)>2):
            image[:,:0] = image[:,:0] * r
            image[:,:1] = image[:,:1]* g
            image[:,:2] = image[:,:2] * b


        image = clip(image.data, 0, 2**16 - 1)
        return image

    @staticmethod
    def wavelet_sharpen(input_image, amount, radius):
        """
        Sharpen a B/W or color image with wavelets. The underlying algorithm was taken from the
        Gimp wavelet plugin, originally written in C and published under the GPLv2+ license at:
        https://github.com/mrossini-ethz/gimp-wavelet-sharpen/blob/master/src/wavelet.c

        :param input_image: Input image (B/W or color), type uint16
        :param amount: Amount of sharpening
        :param radius: Radius in pixels
        :return: Sharpened image, same format as input image
        """

        height, width = input_image.shape[:2]
        color = len(input_image.shape) == 3
        # Allocate workspace: Three complete images, plus 1D object with length max(row, column).
        if color:
            fimg = empty((3, height, width, 3), dtype=float32)
            temp = zeros((max(width, height), 3), dtype=float32)
        else:
            fimg = empty((3, height, width), dtype=float32)
            temp = zeros(max(width, height), dtype=float32)

        # Convert input image to floats.
        if input_image.max()>1:
            fimg[0] = input_image / 65535
        else:
            fimg[0] = input_image

        # Start with level 0. Store its Laplacian on level 1. The operator is separated in a
        # column and a row operator.
        hpass = 0
        for lev in range(5):
            # Highpass and lowpass levels use image indices 1 and 2 in alternating mode to save
            # space.
            lpass = ((lev & 1) + 1)

            if color:
                for row in range(height):
                    AstroImageProcessing.mexican_hat_color(temp, fimg[hpass][row, :, :], width, 1 << lev)
                    fimg[lpass][row, :, :] = temp[:width, :] * 0.25
                for col in range(width):
                    AstroImageProcessing.mexican_hat_color(temp, fimg[lpass][:, col, :], height, 1 << lev)
                    fimg[lpass][:, col, :] = temp[:height, :] * 0.25
            else:
                for row in range(height):
                    AstroImageProcessing.mexican_hat(temp, fimg[hpass][row, :], width, 1 << lev)
                    fimg[lpass][row, :] = temp[:width] * 0.25
                for col in range(width):
                    AstroImageProcessing.mexican_hat(temp, fimg[lpass][:, col], height, 1 << lev)
                    fimg[lpass][:, col] = temp[:height] * 0.25

            # Compute the amount of the correction at the current level.
            amt = amount[lev] * exp(-(lev - radius[lev]) * (lev - radius[lev]) / 1.5) + 1.

            fimg[hpass] -= fimg[lpass]
            fimg[hpass] *= amt

            # Accumulate all corrections in the first workspace image.
            if hpass:
                fimg[0] += fimg[hpass]

            hpass = lpass

        if input_image.max()>1:
            coeff=65535.0
        else:
            coeff=1.0
        # At the end add the coarsest level and convert back to 16bit integer format.
        fimg[0] = ((fimg[0] + fimg[lpass]) * coeff).clip(min=0., max=65535.)
        return fimg[0].astype(uint16)

    @staticmethod
    def mexican_hat(temp, base, size, sc):
        """
        Apply a 1D strided second derivative to a row or column of a B/W image. Store the result
        in the temporary workspace "temp".

        :param temp: Workspace (type float32), length at least "size" elements
        :param base: Input image (B/W), Type float32
        :param size: Length of image row / column
        :param sc: Stride (power of 2) of operator
        :return: -
        """

        # Special case at begin of row/column. Full operator not applicable.
        temp[:sc] = 2 * base[:sc] + base[sc:0:-1] + base[sc:2 * sc]
        # Apply the full operator.
        temp[sc:size - sc] = 2 * base[sc:size - sc] + base[:size - 2 * sc] + base[2 * sc:size]
        # Special case at end of row/column. The full operator is not applicable.
        temp[size - sc:size] = 2 * base[size - sc:size] + base[size - 2 * sc:size - sc] + \
                               base[size - 2:size - 2 - sc:-1]

    @staticmethod
    def mexican_hat_color(temp, base, size, sc):
        """
        Apply a 1D strided second derivative to a row or column of a color image. Store the result
        in the temporary workspace "temp".

        :param temp: Workspace (type float32), length at least "size" elements (first dimension)
                     times 3 colors (second dimension).
        :param base: Input image (color), Type float32
        :param size: Length of image row / column
        :param sc: Stride (power of 2) of operator
        :return: -
        """

        # Special case at begin of row/column. Full operator not applicable.
        temp[:sc, :] = 2 * base[:sc, :] + base[sc:0:-1, :] + base[sc:2 * sc, :]
        # Apply the full operator.
        temp[sc:size - sc, :] = 2 * base[sc:size - sc, :] + base[:size - 2 * sc, :] + base[
                                2 * sc:size, :]
        # Special case at end of row/column. The full operator is not applicable.
        temp[size - sc:size, :] = 2 * base[size - sc:size, :] + base[size - 2 * sc:size - sc, :] + \
                                  base[size - 2:size - 2 - sc:-1, :]


    @staticmethod
    def gaussian_sharpen(input_image, amount, radius, luminance_only=False):
        """
        Sharpen an image with a Gaussian kernel. The input image can be B/W or color.

        :param input_image: Input image, type uint16
        :param amount: Amount of sharpening
        :param radius: Radius of Gaussian kernel (in pixels)
        :param luminance_only: True, if only the luminance channel of a color image is to be
                               sharpened. Default is False.
        :return: The sharpened image (B/W or color, as input), type uint16
        """

        color = len(input_image.shape) == 3

        # Translate the kernel radius into standard deviation.
        sigma = radius / 3

        # Convert the image to floating point format.
        image = input_image.astype(float32)

        # Special case: Only sharpen the luminance channel of a color image.
        if color and luminance_only:
            hsv = cvtColor(image, COLOR_BGR2HSV)
            luminance = hsv[:, :, 2]

            # Apply a Gaussian blur filter, subtract it from the original image, and add a multiple
            # of this correction to the original image. Clip values out of range.
            luminance_blurred = GaussianBlur(luminance, (0, 0), sigma, borderType=BORDER_DEFAULT)
            hsv[:, :, 2] = (luminance + amount * (luminance - luminance_blurred)).clip(min=0.,
                                                                                       max=65535.)
            # Convert the image back to uint16.
            return cvtColor(hsv, COLOR_HSV2BGR).astype(uint16)
        # General case: Treat the entire image (B/W or color 16bit mode).
        else:
            image_blurred = GaussianBlur(image, (0, 0), sigma, borderType=BORDER_DEFAULT)
            return (image + amount * (image - image_blurred)).clip(min=0., max=65535.).astype(
                uint16)

    @staticmethod
    def gaussian_blur(input_image, amount, radius, luminance_only=False):
        """
        Soften an image with a Gaussian kernel. The input image can be B/W or color.

        :param input_image: Input image, type uint16
        :param amount: Amount of blurring, between 0. and 1.
        :param radius: Radius of Gaussian kernel (in pixels)
        :param luminance_only: True, if only the luminance channel of a color image is to be
                               blurred. Default is False.
        :return: The blurred image (B/W or color, as input), type uint16
        """

        color = len(input_image.shape) == 3

        # Translate the kernel radius into standard deviation.
        sigma = radius / 3

        # Convert the image to floating point format.
        image = input_image.astype(float32)

        # Special case: Only blur the luminance channel of a color image.
        if color and luminance_only:
            hsv = cvtColor(image, COLOR_BGR2HSV)
            luminance = hsv[:, :, 2]

            # Apply a Gaussian blur filter, subtract it from the original image, and add a multiple
            # of this correction to the original image. Clip values out of range.
            luminance_blurred = GaussianBlur(luminance, (0, 0), sigma, borderType=BORDER_DEFAULT)
            hsv[:, :, 2] = (luminance_blurred*amount + luminance*(1.-amount)).clip(min=0.,
                                                                                       max=65535.)
            # Convert the image back to uint16.
            return cvtColor(hsv, COLOR_HSV2BGR).astype(uint16)
        # General case: Treat the entire image (B/W or color 16bit mode).
        else:
            image_blurred = GaussianBlur(image, (0, 0), sigma, borderType=BORDER_DEFAULT)
            return (image_blurred*amount + image*(1.-amount)).clip(min=0., max=65535.).astype(
                uint16)

    
    @staticmethod
    def stretch(image, intensity=0.2):
        
        min_val = percentile(image, intensity)
        max_val = percentile(image, 100 - intensity)

        if (min_val!=max_val):
            image = (clip((image - min_val) * (65535.0 / (max_val - min_val) ), 0, 65535)).astype(uint16)

        return image

    @staticmethod
    def image_resize_width(image, new_width):
        h = int(image.shape[0]*new_width/image.shape[1])
        return resize(image, (new_width, h))
    
    @staticmethod
    def image_resize(image, factor):
        return resize(image, (int(image.shape[1]*factor), int(image.shape[0]*factor)))

    @staticmethod
    def fourier_transform(image):
        """ Applique la transformation de Fourier à une image. """
        f_transform = fft.fft2(image)
        f_shift = fft.fftshift(f_transform)
        return f_shift#20*log(f_shift+1)

    @staticmethod
    def calculate_correlation(image, image2):
        # Convertir les images en flottants pour le calcul de corrélation
        image = image.astype(float32)
        image2 = image2.astype(float32)

        # Calculer la corrélation normalisée
        #correlation = cv2.matchTemplate(image, image2, cv2.TM_CCORR_NORMED)
        correlation = correlate2d(image, image2, mode='full', boundary='fill', fillvalue=0)
        return correlation

    @staticmethod
    def to_int8(image):
        return (image/image.max() * 255).astype(uint8)

    @staticmethod
    def get_contour_params(cnt):
        M = cv2.moments(cnt)
        l1 = cnt[:,:,0].max() - cnt[:,:,0].min()
        l2 = cnt[:,:,1].max() - cnt[:,:,1].min()
        l = int(max(l1,l2))
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY, l)

    @staticmethod
    def find_roi(image):
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        thresh = cv2.threshold(blurred, 10000, 65535, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)


        contours = cv2.findContours(AstroImageProcessing.to_int8(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(contours)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        cropped_image = None
        if len(cnts)==2: 

            
            """cX1,cY1,l1 = AstroImageProcessing.get_contour_params(cnts[0])
            cX2,cY2,l2 = AstroImageProcessing.get_contour_params(cnts[1])
            print(cX1,cY1,cX2,cY2,l1,l2)
            cX = int((cX1 + cX2) / 2)
            cY = int((cY1 + cY2) / 2)
            l=max(abs((cX1-l1)-(cX2-l2)),abs((cY1-l1)-(cY2-l2)))
            print(cX,cY,l)"""

            x_min, x_max, y_min, y_max = (inf, -inf, inf, -inf)
            for cnt in cnts:
                (x, y, w, h) = cv2.boundingRect(cnt)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            lx = x_max - x_min
            ly = y_max - y_min
            l = max(lx,ly)


            cropped_image = image[y_min:y_min+l, x_min:x_min+l]
        
        else:
            if len(cnts)==0:
                return None
            cX,cY,l = AstroImageProcessing.get_contour_params(cnts[0])
            cropped_image = image[cY-l:cY+l, cX-l:cX+l]
        return cropped_image
    
    @staticmethod
    def draw_contours(image, level=0.8):
        contours = measure.find_contours(image, level)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=plt.cm.gray)
        thresh = image > percentile(image, 30) 

        regions_max = {}
        for contour in contours:
            rr, cc = polygon(contour[:, 0], contour[:, 1], thresh.shape)
            area = len(rr)  # chaque pixel compte pour une unité d'aire
            if area>3 and len(contour)>20:
                regions_max[area]=contour

        areas = list(regions_max.keys())
        areas.sort(reverse=True)


        for k in areas:
            contour = regions_max[k]
            # Calculer le rectangle englobant et d'autres propriétés
            minr, minc, maxr, maxc = measure.regionprops(contour.astype(int))[0].bbox
            centroid = measure.regionprops(contour.astype(int))[0].centroid
            width = maxc - minc
            height = maxr - minr

            # Afficher les résultats

            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        #plt.show()
        return (contours,fig)
    
    @staticmethod
    def draw_circle(image):
        _, thresh = cv2.threshold(image, 3, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Initialiser la liste pour les coordonnées des centres des petits cercles
        small_circle_coords = []
        for cnt in contours:
            # Calculer le cercle de bord minimal pour chaque contour
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            # Filtrer pour trouver les petits cercles (en supposant que les grands cercles ont un rayon plus grand)
            #if radius < 10:  # Le seuil de rayon peut être ajusté en fonction de la taille de l'image
            small_circle_coords.append(center)

        #for center in small_circle_coords:
        #    cv2.circle(image, center, 1, (255, 0, 0), 2)

        # Afficher les centres des petits cercles
        return image


    @staticmethod
    def autocorrelation(image):
        # Convertir l'image en niveaux de gris si elle est en couleur

        # Normaliser l'image
        image = image.astype(float32) - mean(image)
        image = image / std(image)

        # Calculer l'autocorrelation
        result = correlate2d(image, image, mode='full', boundary='fill', fillvalue=0)

        return (result.astype(uint16)*65535)
    

    @staticmethod
    def inverse_fourier_transform(fourier_image, shift=True):
        # Appliquer la transformée de Fourier inverse
        if shift:
            f_ishift = fft.ifftshift(fourier_image)
        else:
            f_ishift = fourier_image
        img_back = fft.ifftshift(fft.ifft2(f_ishift))

        # Prendre la magnitude pour obtenir l'image réelle
        #img_back = abs(img_back)

        return img_back

    @staticmethod
    def resize_with_padding(image, target_width, target_height):
        h, w = image.shape[:2]
        if h==target_height and w==target_width:
            return image
        delta_w = target_width - w
        delta_h = target_height - h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = 0
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    @staticmethod
    def average_images(images):
        """ Calcule l'image moyenne à partir d'un ensemble d'images de tavelures. """
        mean_images = mean(images, axis=0)
        return (mean_images*65535/mean_images.max()).astype(uint16)

    @staticmethod
    def sum_images(images):
        """ Calcule l'image moyenne à partir d'un ensemble d'images de tavelures. """
        sum = sum(images, axis=0)
        return (sum)
    

    @staticmethod            
    def crosscorr(img1, img2):
        fft_product = (fft.fft2(img1) * fft.fft2(img2).conj())
        cc_data0 = abs(fft.fftshift(fft.ifft2(fft_product)))
        return cc_data0
    
    @staticmethod
    def crosscorfft(im1, im2):
        fft_product = (im1 * im2.conj())
        return fft_product
    
    @staticmethod
    def apply_mean_mask_subtraction(image, k_size=3):

        processed_image = image.copy()
        # Créer un noyau moyen
        kernel = ones((k_size, k_size), float32) / (k_size * k_size)

        # Appliquer le filtre moyenneur
        mean_filtered = cv2.filter2D(image, -1, kernel)

        # Soustraire le résultat filtré de l'image originale
        processed_image = cv2.subtract(processed_image, mean_filtered)
        
        return processed_image