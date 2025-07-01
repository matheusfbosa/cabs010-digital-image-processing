import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


SHOW_IMAGES = True

# Dominância mínima e máxima do verde
GREEN_DOMINANCE_MIN = 0.01
GREEN_DOMINANCE_MAX = 0.25


def green_mask(image_float):
    # Converte a imagem para float
    b, g, r = cv.split(image_float)
    
    # Calcula a dominância do verde
    greenness = g - np.maximum(r, b)
    
    # Normaliza
    alpha = (greenness - GREEN_DOMINANCE_MIN) / (GREEN_DOMINANCE_MAX - GREEN_DOMINANCE_MIN)
    alpha = np.clip(alpha, 0, 1)
    
    # Inverte a máscara e suaviza
    mask = 1.0 - alpha
    mask = cv.GaussianBlur(mask, (3, 3), 0)
    return mask

def pos_process(image_float, mask_float, background_float):    
    # Converte a imagem para HLS (Hue, Lightness, Saturation)    
    image_hls = cv.cvtColor(image_float, cv.COLOR_BGR2HLS)
    luminance = image_hls[:, :, 1]

    # Remove o excesso de verde das áreas de transição (despill)
    b, g, r = cv.split(image_float)
    mask_inv = 1.0 - mask_float

    # Mistura a cor original com a luminância nas áreas de transição
    r = r * mask_float + luminance * mask_inv
    g = g * mask_float + luminance * mask_inv
    b = b * mask_float + luminance * mask_inv
    despilled = cv.merge([b, g, r])

    # Expande a máscara para 3 canais de cor
    mask_3ch = cv.merge([mask_float]*3) 

    # Combina imagem e fundo usando a máscara
    result = despilled * mask_3ch + background_float * (1.0 - mask_3ch)
    
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

        image_float = image.astype(np.float32) / 255.0
        background_float = background.astype(np.float32) / 255.0
        background_resized = cv.resize(background_float, (image.shape[1], image.shape[0]))
        
        mask_float = green_mask(image_float)
        result_float = pos_process(image_float, mask_float, background_resized)
        
        result_uint8 = np.clip(result_float * 255, 0, 255).astype(np.uint8)
        cv.imwrite(f'results/{filename}', result_uint8)

        if SHOW_IMAGES:
            mask_uint8 = (mask_float * 255).astype(np.uint8)
            show_images(image, mask_uint8, result_uint8)


if __name__ == '__main__':
    main()
