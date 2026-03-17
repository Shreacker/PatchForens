import cv2
import scipy
import pywt
import numpy as np
from skimage.transform import rotate
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage import io, color
from skimage.color import label2rgb
from skimage.util import img_as_ubyte
from numpy.fft import fft2, fftshift
from scipy.fftpack import dct

from utils.utilities import *

class Extractors:
    
    def __init__(self, image, patch_size=16, stride=8):
        self.image = image
        self.patch_size = patch_size
        self.stride = stride

    def lbp_extract(
            self,
            P=8,
            R=3,
            multi_radi=True,
            method='uniform'
    ):
        self.image = convert2gray(self.image)
        h, w = self.image.shape
        self.image, h, w = resize4patch(self.image, h, w, self.patch_size)

        if method == 'uniform':
            n_bins = P + 2
        else:
            n_bins = 2 ** P

        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = self.image[y:y+self.patch_size, x:x+self.patch_size]

                if not multi_radi:
                    lbp = local_binary_pattern(patch, P=P, R=R, method=method)

                    hist, _ = np.histogram(
                        lbp.ravel(),
                        bins=n_bins,
                        range=(0, n_bins)
                    )

                    # Normalize
                    hist = hist.astype('float')
                    hist /= (hist.sum() + 1e-7)

                    features.append(hist)
                else:
                    patch_features = []
                    for r in range(1, 4):
                        lbp = local_binary_pattern(
                            patch,
                            P=P,
                            R=r,
                            method=method
                        )

                        hist, _ = np.histogram(
                            lbp.ravel(),
                            bins=n_bins,
                            range=(0, n_bins)
                        )

                        # Normalize
                        hist = hist.astype('float')
                        hist /= (hist.sum() + 1e-8)

                        patch_features.extend(hist)

                    features.append(np.array(patch_features))

        feature_matrix = np.array(features)

        return feature_matrix
    
    def noise_residual_extract(
            self,
            n_bins=32,
            blur_kernel=5,
    ):
        self.image = convert2gray(self.image)
        h, w = self.image.shape
        self.image, h, w = resize4patch(self.image, h, w, self.patch_size)
        residual = computeNoiseRes(self.image, blur_kernel)

        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = residual[y:y+self.patch_size, x:x+self.patch_size]

                hist, _ = np.histogram(
                    patch.ravel(),
                    bins=n_bins,
                    range=(0, 255)
                )
                mean = np.mean(patch)
                std = np.std(patch)
                skewness = skew(patch.ravel())
                kurt = kurtosis(patch.ravel())

                # Normalize
                hist = hist.astype('float')
                hist /= (hist.sum() + 1e-7)

                feature_vector = np.concatenate([hist, [mean, std, skewness, kurt]])
                features.append(feature_vector)

        feature_matrix = np.array(features)

        return feature_matrix
    
    def fft_extract(
            self,
            use_residual=False,
            blur_kernel=5
    ):
        self.image = convert2gray(self.image)
        h, w = self.image.shape
        self.image, h, w = resize4patch(self.image, h, w, self.patch_size)
        if use_residual:
            residual = computeNoiseRes(self.image, blur_kernel)

        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):

                if use_residual:
                    patch = residual[y:y+self.patch_size, x:x+self.patch_size]
                else:
                    patch = self.image[y:y+self.patch_size, x:x+self.patch_size]

                # Compute FFT
                F = fft2(patch)
                F = fftshift(F)
                magnitude = np.abs(F)

                # Power spectrum
                power = magnitude ** 2

                # Create frequency radius grid
                cy, cx = self.patch_size // 2, self.patch_size // 2
                y_grid, x_grid = np.ogrid[:self.patch_size, :self.patch_size]
                radius = np.sqrt((y_grid - cy)**2 + (x_grid - cx)**2)

                r_max = radius.max()

                # Frequency bands
                low = power[radius < r_max/3]
                mid = power[(radius >= r_max/3) & (radius < 2*r_max/3)]
                high = power[radius >= 2*r_max/3]

                low_energy = low.sum()
                mid_energy = mid.sum()
                high_energy = high.sum()

                # Normalize energies
                total_energy = low_energy + mid_energy + high_energy + 1e-8
                low_energy /= total_energy
                mid_energy /= total_energy
                high_energy /= total_energy

                # Spectral entropy
                p = power.ravel()
                p = p / (p.sum() + 1e-8)
                spectral_entropy = -np.sum(p * np.log2(p + 1e-8))

                feature_vector = np.array([
                    low_energy,
                    mid_energy,
                    high_energy,
                    spectral_entropy
                ])
                features.append(feature_vector)

        feature_matrix = np.array(features)
        
        return feature_matrix
    
    def wavelet_extract(
            self,
            wavelet='haar',
            use_residual=False,
            blur_kernel=5
    ):
        self.image = convert2gray(self.image)
        h, w = self.image.shape
        self.image, h, w = resize4patch(self.image, h, w, self.patch_size)
        if use_residual:
            residual = computeNoiseRes(self.image, blur_kernel)

        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):

                if use_residual:
                    patch = residual[y:y+self.patch_size, x:x+self.patch_size]
                else:
                    patch = self.image[y:y+self.patch_size, x:x+self.patch_size]

                coeffs2 = pywt.dwt2(patch, wavelet)
                LL, (LH, HL, HH) = coeffs2

                band_features = []
                for band in [LH, HL, HH]:
                    mean = np.mean(band)
                    std = np.std(band)
                    energy = np.sum(band**2)
                    kurt = kurtosis(band.ravel())

                    band_features.extend([mean, std, energy, kurt])
                features.append(np.array(band_features))

        feature_matrix = np.array(features)

        return feature_matrix
    
    def dct_extract(
            self,
            use_residual=False,
            blur_kernel=5,
    ):
        self.image = convert2gray(self.image)
        h, w = self.image.shape
        self.image, h, w = resize4patch(self.image, h, w, self.patch_size)
        if use_residual:
            residual = computeNoiseRes(self.image, blur_kernel)

        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):

                if use_residual:
                    patch = residual[y:y+self.patch_size, x:x+self.patch_size]
                else:
                    patch = self.image[y:y+self.patch_size, x:x+self.patch_size]

                # Compute DCT
                D = dct(dct(patch.T, norm='ortho').T, norm='ortho')

                # Power spectrum
                power = D**2

                # Frequency zones
                low = power[:4, :4]
                mid = power[4:8, 4:8]
                high = power[8:, 8:]

                low_energy = low.sum()
                mid_energy = mid.sum()
                high_energy = high.sum()

                total = low_energy + mid_energy + high_energy + 1e-8

                # Normalize energies
                low_energy /= total
                mid_energy /= total
                high_energy /= total

                dc = np.abs(D[0, 0])
                ac_kurt = kurtosis(D[1:, 1:].ravel())

                feature_vector = np.array([
                    low_energy,
                    mid_energy,
                    high_energy,
                    dc,
                    ac_kurt
                ])
                features.append(feature_vector)

        feature_matrix = np.array(features)

        return feature_matrix
    
    def glcm_extract(
            self,
            levels=128,
            use_residual=False,
            blur_kernel=5
    ):
        self.image = convert2gray(self.image)
        h, w = self.image.shape
        self.image, h, w = resize4patch(self.image, h, w, self.patch_size)
        if use_residual:
            residual = computeNoiseRes(self.image, blur_kernel)

        shift = 256 // levels
        self.image = (self.image // shift).astype(np.uint8)
        self.image = np.clip(self.image, 0, levels - 1)
        
        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):

                if use_residual:
                    patch = residual[y:y+self.patch_size, x:x+self.patch_size]
                else:
                    patch = self.image[y:y+self.patch_size, x:x+self.patch_size]
                
                glcm = graycomatrix(
                    patch,
                    distances=[1],
                    angles=[0], # [0, np.pi/4, np.pi/2, 3*np.pi/4],
                    levels = levels,
                    symmetric=True,
                    normed=True
                )

                corre = graycoprops(glcm, 'correlation').mean()
                con = graycoprops(glcm, 'contrast').mean()
                eng = graycoprops(glcm, 'energy').mean()
                homo = graycoprops(glcm, 'homogeneity').mean()

                features.append([con, corre, eng, homo])
            
        feature_matrix = np.array(features)

        return feature_matrix
    
    def stat_extract(
            self,
    ):
        self.image = convert2gray(self.image)
        h, w = self.image.shape
        self.image, h, w = resize4patch(self.image, h, w, self.patch_size)

        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = self.image[y:y+self.patch_size, x:x+self.patch_size]
                patch_1d = patch.flatten()

                kurt = kurtosis(patch_1d, fisher=True)
                skewness = skew(patch_1d)
                smooth = 1 - (1 / (1 + np.std(patch_1d)**2))
                corr = corrcoef(patch_1d)

                features.append([kurt, skewness, corr, smooth])

        feature_matrix = np.array(features)

        return feature_matrix