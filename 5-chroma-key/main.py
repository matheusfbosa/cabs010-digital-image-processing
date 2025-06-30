import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


SHOW_IMAGES = True

# Dominância mínima e máxima do verde
GREEN_DOMINANCE_MIN = 0.01
GREEN_DOMINANCE_MAX = 0.25

# Coeficientes para calcular a luminância da imagem
LUMINANCE_COEF_BLUE = 0.114
LUMINANCE_COEF_GREEN = 0.587
LUMINANCE_COEF_RED = 0.299


def green_mask(image):
    # Converte a imagem para float
    image_float = image.astype(np.float32) / 255.0
    b, g, r = cv.split(image_float)
    
    # Calcula a dominância do verde
    greenness = g - np.maximum(r, b)
    
    # Normaliza
    alpha = (greenness - GREEN_DOMINANCE_MIN) / (GREEN_DOMINANCE_MAX - GREEN_DOMINANCE_MIN)
    alpha = np.clip(alpha, 0, 1)
    
    # Inverte a máscara e suaviza
    mask = (1.0 - alpha) * 255
    mask = mask.astype(np.uint8)
    mask = cv.GaussianBlur(mask, (3, 3), 0)
    return mask

def pos_process(image, mask, background):
    # Converte as imagens para float
    image_float = image.astype(np.float32)
    mask_float = mask.astype(np.float32) / 255.0
    
    b, g, r = cv.split(image_float)

    # Calcula a luminância
    luminance = (LUMINANCE_COEF_BLUE * b +
                 LUMINANCE_COEF_GREEN * g +
                 LUMINANCE_COEF_RED * r)
    
    # Remove o excesso de verde das áreas de transição
    mask_inv = 1.0 - mask_float
    r = r * mask_float + luminance * mask_inv
    g = g * mask_float + luminance * mask_inv
    b = b * mask_float + luminance * mask_inv
    despilled = cv.merge([b, g, r])

    background_float = background.astype(np.float32)
    
    # Expande a máscara para 3 canais de cor
    mask_3ch = cv.merge([mask_float]*3) 

    # Combina imagem e fundo usando a máscara
    result = despilled * mask_3ch + background_float * (1.0 - mask_3ch)
    
    # Normaliza
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def show_images(image, mask, result):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Máscara')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    plt.title('Chroma Key')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    background = cv.imread('img/background.bmp')

    # Processa as imagens de 0.bmp a 7.bmp
    for i in range(0, 8):
        filename = f'{i}.bmp'
        image = cv.imread(f'img/{filename}')

        background_resized = cv.resize(background, (image.shape[1], image.shape[0]))
        mask = green_mask(image)
        chroma_key = pos_process(image, mask, background_resized)

        cv.imwrite(f'results/{filename}', chroma_key)

        if SHOW_IMAGES:
            show_images(image, mask, chroma_key)


if __name__ == '__main__':
    main()
