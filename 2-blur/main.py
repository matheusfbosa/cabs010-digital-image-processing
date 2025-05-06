import timeit
import numpy as np
import cv2 as cv


INPUT_IMAGE = 'original.bmp'
KERNEL_SIZE = 7


def mean_filter_naive(img, kernel_size):
    img_out = np.zeros_like(img)
    pad = kernel_size // 2

    # Adiciona uma borda de pixels com valor 0 ao redor da imagem
    img_padded = np.pad(img, pad_width=((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    for channel in range(img.shape[2]):
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                img_out[row, col, channel] = np.mean(img_padded[row:row + kernel_size, col:col + kernel_size, channel])

    return img_out


def mean_filter_separable(img, kernel_size):
    img_out = np.zeros_like(img)
    img_out_temp = np.zeros_like(img)
    pad = kernel_size // 2

    # Adiciona uma borda de pixels com valor 0 ao redor da imagem
    img_padded_h = np.pad(img, pad_width=((0, 0), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    # Filtro horizontal    
    for channel in range(img.shape[2]):
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                img_out_temp[row, col, channel] = np.mean(img_padded_h[row, col:col + kernel_size, channel])

    # Adiciona uma borda de pixels com valor 0 ao redor da imagem
    img_padded_v = np.pad(img_out_temp, pad_width=((pad, pad), (0, 0), (0, 0)), mode='constant', constant_values=0)
    
    # Filtro vertical
    for channel in range(img.shape[2]):
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                img_out[row, col, channel] = np.mean(img_padded_v[row:row + kernel_size, col, channel])

    return img_out


def mean_filter_integral_image(img, kernel_size):
    img_out = np.zeros_like(img)
    n_rows, n_cols, n_channels = img.shape
    pad = kernel_size // 2

    img_padded = np.pad(img, pad_width=((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    # Calcula a imagem integral
    integral_image = calculate_integral_image(img_padded)

    for channel in range(n_channels):
        for row in range(n_rows):
            for col in range(n_cols):
                # Define a região de interesse
                start_row = max(0, row - pad)
                start_col = max(0, col - pad)
                end_row = min(n_rows + pad - 1, row + pad)
                end_col = min(n_cols + pad - 1, col + pad)

                # Calcula a soma da região usando a imagem integral (vértice inferior direito)
                region_sum = integral_image[end_row, end_col, channel]
                if start_col > 0:
                    region_sum -= integral_image[end_row, start_col-1, channel]
                if start_row > 0:
                    region_sum -= integral_image[start_row-1, end_col, channel]
                if start_col > 0 and start_row > 0:
                    region_sum += integral_image[start_row-1, start_col-1, channel]

                area = (end_col - start_col + 1) * (end_row - start_row + 1)
                img_out[row, col, channel] = region_sum / area

    return img_out


def calculate_integral_image(img):
    n_rows, n_cols, n_channels = img.shape
    integral_image = np.zeros_like(img)

    for channel in range(n_channels):
        for row in range(n_rows):
            for col in range(n_cols):
                left_sum = integral_image[row, col-1, channel] if col > 0 else 0
                top_sum = integral_image[row-1, col, channel] if row > 0 else 0
                diagonal_sum = integral_image[row-1, col-1, channel] if col > 0 and row > 0 else 0
                integral_image[row, col, channel] = img[row, col, channel] + left_sum + top_sum - diagonal_sum

    return integral_image


def compare_images(img1, img2):
    diff = np.abs(img1 - img2)
    print(f'Max diff: {np.max(diff):.3f}')
    print(f'Min diff: {np.min(diff):.3f}')


def main():
    img = cv.imread(INPUT_IMAGE, cv.IMREAD_GRAYSCALE)
    assert img is not None, 'image could not be read'

    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype(np.float32) / 255
    cv.imshow('original', img)

    # Imagem gerada com OpenCV para comparação (borda com valor constante de 0)
    img_out_cv = cv.blur(img, ksize=(KERNEL_SIZE, KERNEL_SIZE), borderType=cv.BORDER_CONSTANT)
    img_out_cv = img_out_cv.reshape((img_out_cv.shape[0], img_out_cv.shape[1], 1))
    cv.imshow('opencv', img_out_cv)
    cv.imwrite('opencv.png', img_out_cv*255)

    # Filtro da média ingênuo
    start_time = timeit.default_timer()
    img_out_naive = mean_filter_naive(img, kernel_size=KERNEL_SIZE)
    print('[Naive] Duration: %f' % (timeit.default_timer() - start_time))
    compare_images(img1=img_out_naive, img2=img_out_cv)
    cv.imshow('naive', img_out_naive)
    cv.imwrite('naive.png', img_out_naive*255)

    # Filtro da média separável
    start_time = timeit.default_timer()
    img_out_separable = mean_filter_separable(img, kernel_size=KERNEL_SIZE)
    print('[Separable] Duration: %f' % (timeit.default_timer() - start_time))
    compare_images(img1=img_out_separable, img2=img_out_cv)
    cv.imshow('separable', img_out_separable)
    cv.imwrite('separable.png', img_out_separable*255)

    # Filtro da média com imagem integral
    start_time = timeit.default_timer()
    img_out_integral = mean_filter_integral_image(img, kernel_size=KERNEL_SIZE)
    print('[Integral] Duration: %f' % (timeit.default_timer() - start_time))
    compare_images(img1=img_out_separable, img2=img_out_cv)
    cv.imshow('integral', img_out_integral)
    cv.imwrite('integral.png', img_out_integral*255)

    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
