import cv2
import numpy as np


# IMAGE_INPUT = 'GT2.BMP'
IMAGE_INPUT = 'Wind Waker GC.bmp'

ALPHA = 1.0
BETA = 0.5
GAMMA = 0
SIGMAS = [5, 10, 20]
BLOOM_BOX_KERNEL_SIZE = 15
BLOOM_BOX_ITERATIONS = 3


def bright_pass_filter(img, threshold=180):
    # Converte imagem para escala de cinza
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Cria máscara com somente valores acima do threshold
    mask = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_TOZERO)[1]

    # Expande máscara para 3 canais
    mask_3c = cv2.merge([mask]*3)
    img_bright_pass = cv2.bitwise_and(img, mask_3c)

    cv2.imwrite('bright-pass.jpg', img_bright_pass)    
    return img_bright_pass

def apply_gaussian_bloom(img, img_bright_pass, sigmas=[5, 10, 20], alpha=1.0, beta=0.0, gamma=0):    
    img_blurred = np.zeros_like(img, dtype=np.float32)
    
    # Aplica filtro gaussiano com diferentes sigmas
    for sigma in sigmas:
        blurred = cv2.GaussianBlur(img_bright_pass, (0, 0), sigmaX=sigma, sigmaY=sigma)
        img_blurred += blurred.astype(np.float32)

    # Normaliza imagem
    img_blurred = np.clip(img_blurred, 0, 255).astype(np.uint8)

    # Adiciona imagem borrada à imagem original com soma ponderada
    return cv2.addWeighted(img, alpha, img_blurred, beta, gamma)


def apply_box_bloom(img, img_bright_pass, kernel_size=15, iterations=3, alpha=1.0, beta=0.0, gamma=0):
    img_blurred = img_bright_pass.copy().astype(np.float32)

    # Aplica filtro da média sucessivamente
    for _ in range(iterations):
        img_blurred = cv2.blur(img_blurred, (kernel_size, kernel_size))

    # Normaliza imagem
    img_blurred = np.clip(img_blurred, 0, 255).astype(np.uint8)

    # Adciona imagem borrada à imagem original com soma ponderada
    return cv2.addWeighted(img, alpha, img_blurred, beta, gamma)

def main():
    img = cv2.imread(IMAGE_INPUT)
    assert img is not None, f'image {IMAGE_INPUT} not found'
    cv2.imshow('original', img)

    # Bright Pass
    img_bright_pass = bright_pass_filter(img)
    cv2.imshow('bright-pass', img_bright_pass)
    cv2.imwrite(f'{IMAGE_INPUT} - bright-pass.jpg', img_bright_pass)

    # Bloom Gaussian
    img_bloom_gaussian = apply_gaussian_bloom(
        img,
        img_bright_pass,
        sigmas=SIGMAS,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
    )
    cv2.imshow('bloom-gaussian', img_bloom_gaussian)
    cv2.imwrite(f'{IMAGE_INPUT} - bloom-gaussian.jpg', img_bloom_gaussian)

    # Bloom Box com filtro da média
    img_bloom_box = apply_box_bloom(
        img,
        img_bright_pass,
        kernel_size=BLOOM_BOX_KERNEL_SIZE,
        iterations=BLOOM_BOX_ITERATIONS,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
    )
    cv2.imshow('bloom-box', img_bloom_gaussian)
    cv2.imwrite(f'{IMAGE_INPUT} - bloom-box.jpg', img_bloom_box)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
