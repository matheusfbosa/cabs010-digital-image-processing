import cv2
import numpy as np


# IMAGE_INPUT = 'GT2.BMP'
IMAGE_INPUT = 'Wind Waker GC.bmp'

ALPHA = 1.0
BETA = 0.5
GAMMA = 0
SIGMAS = [5, 10, 20]
BLOOM_BOX_KERNELS_SIZE = [15, 31, 45]
BLOOM_BOX_ITERATIONS = 3


def bright_pass_filter(img, threshold=180):
    # Converte a imagem para escala de cinza para facilitar a criação da máscara
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Cria uma máscara binária
    # Pixels na imagem em escala de cinza com valor maior que o 'threshold' se tornam 255 (branco), e os outros se tornam 0 (preto)
    _, mask = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)

    # Expande a máscara de 1 canal (preto e branco) para 3 canais (RGB) para que seja compatível com a imagem colorida original
    mask_3c = cv2.merge([mask, mask, mask])

    # Usa a operação 'E' bit-a-bit (bitwise AND)
    # Onde a máscara é branca (255), os pixels da imagem original são mantidos
    # Onde a máscara é preta (0), os pixels da imagem original se tornam pretos, efetivamente "filtrando" as áreas escuras
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

def apply_box_bloom(img, img_bright_pass, kernel_sizes=[15, 31, 45], iterations=3, alpha=1.0, beta=1.0, gamma=0):
    # Converte a imagem de entrada para float para evitar perda de precisão durante as somas
    img_float = img.copy().astype(np.float32)
    blur_total = np.zeros_like(img_float)

    # Itera sobre os diferentes tamanhos de kernel para simular sigmas maiores
    for ksize in kernel_sizes:
        # A cada iteração de ksize, começa o blur a partir do bright-pass original
        img_blurred = img_bright_pass.copy().astype(np.float32)
        
        # Aplica o filtro da média (box blur) sucessivamente
        for _ in range(iterations):
            img_blurred = cv2.blur(img_blurred, (ksize, ksize))
            
        # Acumula o resultado de cada passe de blur
        blur_total += img_blurred

    # Normaliza imagem
    bloom_effect = np.clip(blur_total, 0, 255).astype(np.uint8)

    # Adiciona imagem borrada à imagem original com soma ponderada
    return cv2.addWeighted(img, alpha, bloom_effect, beta, gamma)

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
        kernel_sizes=BLOOM_BOX_KERNELS_SIZE,
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
