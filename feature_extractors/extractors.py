import cv2
import scipy
import pywt
import numpy as np
from skimage.transform import rotate
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from skimage import io, color
from skimage.color import label2rgb
from skimage.util import img_as_ubyte
from numpy.fft import fft2, fftshift
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fftpack import dct
# from scipy.stats import skew, kurtosis

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
        image = self.image.copy()
        image = convert2gray(image)
        h, w = image.shape
        image, h, w = resize4patch(image, h, w, self.patch_size)

        if method == 'uniform':
            n_bins = P + 2
        else:
            n_bins = 2 ** P

        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]

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
            noise_range: list | tuple = [-30, 30]
    ):
        if not isinstance(noise_range, (list, tuple)) or len(noise_range) != 2:
            raise ValueError('noise_range must be a list or a tuple of length 2')

        image = self.image.copy()
        image = convert2gray(image)
        h, w = image.shape
        image, h, w = resize4patch(image, h, w, self.patch_size)
        residual = computeNoiseRes(image, blur_kernel)
        residual = np.clip(residual, noise_range[0], noise_range[1])

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
        image = self.image.copy()
        image = convert2gray(image)
        h, w = image.shape
        image, h, w = resize4patch(image, h, w, self.patch_size)
        if use_residual:
            residual = computeNoiseRes(image, blur_kernel)

        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):

                if use_residual:
                    patch = residual[y:y+self.patch_size, x:x+self.patch_size]
                else:
                    patch = image[y:y+self.patch_size, x:x+self.patch_size]

                # Smoothen edges
                patch = hannWindow(patch, self.patch_size)

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
        image = self.image.copy()
        image = convert2gray(image)
        h, w = image.shape
        image, h, w = resize4patch(image, h, w, self.patch_size)
        if use_residual:
            residual = computeNoiseRes(image, blur_kernel)

        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):

                if use_residual:
                    patch = residual[y:y+self.patch_size, x:x+self.patch_size]
                else:
                    patch = image[y:y+self.patch_size, x:x+self.patch_size]

                # Smoothen edges
                patch = hannWindow(patch, self.patch_size)

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
        image = self.image.copy()
        image = convert2gray(image)
        h, w = image.shape
        image, h, w = resize4patch(image, h, w, self.patch_size)
        if use_residual:
            residual = computeNoiseRes(image, blur_kernel)

        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):

                if use_residual:
                    patch = residual[y:y+self.patch_size, x:x+self.patch_size]
                else:
                    patch = image[y:y+self.patch_size, x:x+self.patch_size]

                # Smoothen edges
                patch = hannWindow(patch, self.patch_size)

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
        image = self.image.copy()
        image = convert2gray(image)
        h, w = image.shape
        image, h, w = resize4patch(image, h, w, self.patch_size)
        if use_residual:
            residual = computeNoiseRes(image, blur_kernel)

        shift = 256 // levels
        image = (image // shift).astype(np.uint8)
        image = np.clip(image, 0, levels - 1)
        
        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):

                if use_residual:
                    patch = residual[y:y+self.patch_size, x:x+self.patch_size]
                else:
                    patch = image[y:y+self.patch_size, x:x+self.patch_size]
                
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
        image = self.image.copy()
        image = convert2gray(image)
        h, w = image.shape
        image, h, w = resize4patch(image, h, w, self.patch_size)

        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patch_1d = patch.flatten()

                kurt = kurtosis(patch_1d)
                skewness = skew(patch_1d)
                smooth = 1 - (1 / (1 + np.std(patch_1d)**2))
                corr = corrcoef(patch_1d)

                features.append([kurt, skewness, corr, smooth])

        feature_matrix = np.array(features)

        return feature_matrix
    
    def gsm_extract(
            self,
    ):
        image = self.image.copy()
        image = convert2gray(image)
        h, w = image.shape
        image, h, w = resize4patch(image, h, w, self.patch_size)

        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                
                gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

                gx = np.abs(gx).flatten()
                gy = np.abs(gy).flatten()

                mag = np.sqrt(gx**2 + gy**2).flatten()

                grad_features = []
                for arr in [gx, gy, mag]:
                    grad_features.extend([
                        np.mean(arr),
                        np.std(arr),
                        skew(arr),
                        kurtosis(arr)
                    ])
                features.append(np.array(grad_features))

        feature_matrix = np.array(features)

        return feature_matrix
    
    def laplacian_stat_extract(
            self,
    ):
        image = self.image.copy()
        image = convert2gray(image)
        h, w = image.shape
        image, h, w = resize4patch(image, h, w, self.patch_size)

        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]

                lap = cv2.Laplacian(patch, cv2.CV_64F, borderType=cv2.BORDER_REFLECT)
                lap_abs = np.abs(lap).flatten()

                features.append([
                    lap_abs.mean(),
                    lap_abs.std(),
                    lap_abs.var(),
                    skew(lap_abs),
                    kurtosis(lap_abs)
                ])
        
        feature_matrix = np.array(features)

        return feature_matrix
    
    def hprb_extract(
            self,
            block_size=8,
            stride=4,
            blur_kernel=5,
            noise_range: list | tuple = [-30, 30]
    ):
        if not isinstance(noise_range, (list, tuple)) or len(noise_range) != 2:
            raise ValueError('noise_range must be a list or a tuple of length 2')

        image = self.image.copy()
        image = convert2gray(image)
        h, w = image.shape
        image, h, w = resize4patch(image, h, w, self.patch_size)

        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]

                residual = computeNoiseRes(patch, blur_kernel)
                residual = np.clip(residual, noise_range[0], noise_range[1])

                variances=[]
                hp, wp = residual.shape
                for yp in range(0, hp - block_size + 1, stride):
                    for xp in range(0, wp - block_size + 1, stride):
                        block = residual[yp:yp+block_size, xp:xp+block_size]
                        variances.append(np.var(block))

                variances = np.array(variances)
                features.append([
                    np.mean(variances),
                    np.std(variances),
                    skew(variances),
                    kurtosis(variances)
                ])

        feature_matrix = np.array(features)

        return feature_matrix
    
    def _ensure_3channel_bgr(self, image):
        """Chuẩn hóa ảnh về 3 kênh BGR, xử lý các trường hợp kênh khác nhau"""
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.ndim == 3:
            c = image.shape[2]
            if c == 3:
                return image
            elif c == 4:
                # Có alpha -> bỏ kênh alpha (giả định thứ tự BGRA, là format mặc định của cv2.imread/cv2 pipeline)
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            elif c == 1:
                return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
            else:
                raise ValueError(f"Unsupported number of channels: {c}")

        raise ValueError(f"Unsupported image shape: {image.shape}")

    def hog_extract(
            self,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            blur_kernel=5
    ):
        image = self._ensure_3channel_bgr(self.image.copy())

        image = convert2gray(image)
        h, w = image.shape
        image, h, w = resize4patch(image, h, w, self.patch_size)

        if blur_kernel > 0:
            try:
                image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
            except cv2.error as e:
                raise ValueError(f"GaussianBlur failed (blur_kernel={blur_kernel} must be odd & >0): {e}")

        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                try:
                    patch_hog = hog(
                        patch,
                        orientations=orientations,
                        pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block,
                        feature_vector=True,
                    )
                except Exception as e:
                    # patch quá nhỏ so với pixels_per_cell/cells_per_block sẽ rơi vào đây
                    expected_len = self._hog_feature_len(orientations, pixels_per_cell, cells_per_block)
                    patch_hog = np.zeros(expected_len, dtype=np.float32)
                    print(f"[hog_extract] Warning: patch at ({x},{y}) failed ({e}), filled with zeros")
                features.append(patch_hog)

        if not features:
            return np.empty((0,))

        feature_matrix = np.array(features)
        return feature_matrix

    def _hog_feature_len(self, orientations, pixels_per_cell, cells_per_block):
        """Tính số chiều HOG feature kỳ vọng để điền zero-vector khi patch lỗi"""
        cells_y = self.patch_size // pixels_per_cell[0]
        cells_x = self.patch_size // pixels_per_cell[1]
        blocks_y = max(cells_y - cells_per_block[0] + 1, 0)
        blocks_x = max(cells_x - cells_per_block[1] + 1, 0)

        return blocks_y * blocks_x * cells_per_block[0] * cells_per_block[1] * orientations

    def color_hist_extract(self):
        image = self._ensure_3channel_bgr(self.image.copy())

        h, w, _ = image.shape
        image, h, w = resize4patch(image, h, w, self.patch_size)

        bins = 64
        zero_feat = np.zeros(bins * 3, dtype=np.float32)

        features = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]

                try:
                    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray_patch, 254, 255, cv2.THRESH_BINARY_INV)

                    # Nếu mask rỗng (patch toàn trắng/gần trắng) -> histogram vô nghĩa
                    if cv2.countNonZero(mask) == 0:
                        features.append(zero_feat.copy())
                        continue

                    hsv_white = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                    hist_h = cv2.calcHist([hsv_white], [0], mask, [bins], [0, 180])
                    hist_s = cv2.calcHist([hsv_white], [1], mask, [bins], [0, 256])
                    hist_v = cv2.calcHist([hsv_white], [2], mask, [bins], [0, 256])

                    cv2.normalize(hist_h, hist_h)
                    cv2.normalize(hist_s, hist_s)
                    cv2.normalize(hist_v, hist_v)
                    color_hist_feature = np.concatenate(
                        [hist_h.flatten(), hist_s.flatten(), hist_v.flatten()]
                    )
                    # Phòng trường hợp normalize sinh NaN/Inf
                    color_hist_feature = np.nan_to_num(color_hist_feature, nan=0.0, posinf=0.0, neginf=0.0)

                except cv2.error as e:
                    print(f"[color_hist_extract] Warning: patch at ({x},{y}) failed ({e}), filled with zeros")
                    color_hist_feature = zero_feat.copy()

                features.append(color_hist_feature)

        if not features:
            return np.empty((0,))

        feature_matrix = np.array(features)
        return feature_matrix

    def chroma_correlation_extract(self):
        image = self._ensure_3channel_bgr(self.image.copy())
        h, w, _ = image.shape
        image, h, w = resize4patch(image, h, w, self.patch_size)
        image_ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float64)

        y_ch = image_ycbcr[:, :, 0]
        cr_ch = image_ycbcr[:, :, 1]
        cb_ch = image_ycbcr[:, :, 2]

        # Tạo tất cả các patch dạng view, không copy dữ liệu
        y_windows = sliding_window_view(y_ch, (self.patch_size, self.patch_size))[::self.stride, ::self.stride]
        cb_windows = sliding_window_view(cb_ch, (self.patch_size, self.patch_size))[::self.stride, ::self.stride]
        cr_windows = sliding_window_view(cr_ch, (self.patch_size, self.patch_size))[::self.stride, ::self.stride]

        n_y, n_x = y_windows.shape[:2]
        y_flat = y_windows.reshape(n_y * n_x, -1)
        cb_flat = cb_windows.reshape(n_y * n_x, -1)
        cr_flat = cr_windows.reshape(n_y * n_x, -1)

        std_cb = cb_flat.std(axis=1)
        std_cr = cr_flat.std(axis=1)

        def batched_corr(a, b):
            a_c = a - a.mean(axis=1, keepdims=True)
            b_c = b - b.mean(axis=1, keepdims=True)
            num = (a_c * b_c).sum(axis=1)
            denom = np.sqrt((a_c**2).sum(axis=1) * (b_c**2).sum(axis=1))
            with np.errstate(invalid='ignore', divide='ignore'):
                corr = np.where(denom == 0, 0.0, num / denom)
            return corr

        corr_y_cb = batched_corr(y_flat, cb_flat)
        corr_y_cr = batched_corr(y_flat, cr_flat)
        corr_cb_cr = batched_corr(cb_flat, cr_flat)

        feature_matrix = np.stack([std_cb, std_cr, corr_y_cb, corr_y_cr, corr_cb_cr], axis=1)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_matrix