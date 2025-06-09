import timeit
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import cm


IMAGE_FILENAMES = [
    '60.bmp',
    '82.bmp',
    '114.bmp',
    '150.bmp',
    '205.bmp',
]

SHOW_IMAGES = True
COLORS = [tuple(int(x * 255) for x in cm.tab20(i)[:3]) for i in range(20)]

MIN_BLOB_AREA = 10              # Área mínima em pixels para um blob ser considerado válido
BLOB_SIZE_THRESHOLD = 2         # Número de desvios padrão acima da média para considerar como blob grande
DEBRIS_THRESHOLD = 0.15         # Componentes menores que 15% da média são considerados resíduos
STD_THRESHOLD = 0.3             # Desvio padrão deve ser menor que 30% da média para considerar boa estimativa
SINGLE_GRAIN_THRESHOLD = 1.65   # Componentes menores que 1.65x o tamanho médio são considerados grãos únicos
OVERLAP_FACTOR = 1.7            # Fator de ajuste empírico para sobreposição de grãos


def load_image(image_filename):
    image = cv.imread(f'images/{image_filename}')
    assert image is not None, f'image {image_filename} not found'
    return image

def binarize(image):
    # Como tratar diferentes níveis de iluminação nas imagens?
    # Kernel grande para estimativa de fundo
    image_background = cv.GaussianBlur(image, (51, 51), 0)
    # Subtrai fundo para normalizar iluminação
    image_normalized = cv.subtract(image, image_background)

    # Suavização com filtro gaussiano
    image_blur = cv.GaussianBlur(image_normalized, (7, 7), 0)

    # Binarização com Otsu
    _, image_thresh = cv.threshold(image_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Operação morfológica de abertura
    kernel = np.ones((3, 3), np.uint8)    
    image_thresh = cv.morphologyEx(image_thresh, cv.MORPH_OPEN, kernel, iterations=1)

    return image_thresh

def find_connected_components(image_thresh, image_input_gray):
    # Encontra componentes conexos
    num_labels, labels = cv.connectedComponents(image_thresh)
    
    # Calcula áreas dos componentes
    areas = []
    for label in range(1, num_labels):
        area = np.sum(labels == label)
        if area >= MIN_BLOB_AREA:
            areas.append(area)
    
    # Se não houver componentes, apenas retorna imagem original colorida
    if not areas:
        return cv.cvtColor(image_input_gray, cv.COLOR_GRAY2BGR), 0
    
    # Remove componentes muito pequenos
    mean_area = np.mean(areas)
    areas = [area for area in areas if area >= DEBRIS_THRESHOLD * mean_area]
    
    # Refina a estimativa do tamanho de um grão
    while len(areas) > 2:
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        
        # Se o desvio padrão for menor que X% da média, temos uma boa estimativa
        if std_area < STD_THRESHOLD * mean_area:
            break
            
        # Remove o maior e o menor componente
        areas.remove(max(areas))
        areas.remove(min(areas))
    
    # Tamanho estimado de um grão
    grain_size = mean_area
    
    filtered_labels = labels.copy()
    valid_components = 0
    new_labels = np.zeros_like(labels)
    current_label = 1
    
    # Lista para armazenar os centros dos componentes e número de grãos estimados
    component_centers = []
    
    for label in range(1, num_labels):
        area = np.sum(labels == label)

        if area < MIN_BLOB_AREA:
            filtered_labels[labels == label] = 0
            continue

        # Se a área é menor que X vezes o tamanho estimado de um grão, conta como 1
        if area < SINGLE_GRAIN_THRESHOLD * grain_size:
            new_labels[labels == label] = current_label

            # Calcula o centro do componente
            y_coords, x_coords = np.where(labels == label)
            center_y = int(np.mean(y_coords))
            center_x = int(np.mean(x_coords))
            component_centers.append((center_x, center_y, current_label, 1)) # 1 grão estimado
            current_label += 1
            valid_components += 1
        else:
            # Estima o número de grãos considerando sobreposição
            estimated_grains = max(1, round((area / grain_size) * OVERLAP_FACTOR))
            
            # Calcula o centro do blob
            y_coords, x_coords = np.where(labels == label)
            center_y = int(np.mean(y_coords))
            center_x = int(np.mean(x_coords))
            
            # Adiciona o blob original com o número estimado
            new_labels[labels == label] = current_label
            component_centers.append((center_x, center_y, current_label, estimated_grains))
            current_label += 1
            valid_components += estimated_grains

    # Desenha cada componente com uma cor e adiciona o número
    image_components = cv.cvtColor(image_input_gray, cv.COLOR_GRAY2BGR)
    if valid_components > 0:
        for label in range(1, current_label):
            if np.any(new_labels == label):
                color = COLORS[(label - 1) % len(COLORS)]
                image_components[new_labels == label] = color
        
        # Adiciona os números de grãos estimados
        for center_x, center_y, label, estimated_grains in component_centers:
            cv.putText(image_components, str(estimated_grains), (center_x, center_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image_components, valid_components

def plot_images(images_input, images_thresh, images_components, num_components):
    n_images = len(images_input)
    assert n_images > 0
    assert n_images == len(images_thresh) == len(images_components)
    
    for i in range(n_images):
        # Extrai contagem verdadeira do nome do arquivo
        true_count = int(IMAGE_FILENAMES[i].split('.')[0])     

        plt.figure(figsize=(20, 6))
        
        # Plota imagem original
        plt.subplot(1, 3, 1)
        plt.imshow(images_input[i])
        plt.title(f'Original {IMAGE_FILENAMES[i]}', fontsize=12)
        plt.axis('off')
        
        # Plota imagem binarizada
        plt.subplot(1, 3, 2)
        plt.imshow(images_thresh[i], cmap='gray')
        plt.title(f'Binarizada {IMAGE_FILENAMES[i]}', fontsize=12)
        plt.axis('off')

        # Plota imagem com componentes conexos
        plt.subplot(1, 3, 3)
        plt.imshow(images_components[i])
        plt.title(f'Componentes {IMAGE_FILENAMES[i]}\n{num_components[i]}/{true_count}', fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    images_input = []
    images_thresh = []
    images_components = []
    n_components = []
    total_error = 0
    total_percent_error = 0

    for image_filename in IMAGE_FILENAMES:
        print(f'[{image_filename}] Processando imagem...')
        start_time = timeit.default_timer()

        # Extrai contagem verdadeira do nome do arquivo
        true_count = int(image_filename.split('.')[0])

        # Carrega imagem
        image_input = load_image(image_filename)
        images_input.append(image_input)

        # Binariza imagem
        image_input_gray = cv.cvtColor(image_input, cv.COLOR_BGR2GRAY)
        image_thresh = binarize(image_input_gray)
        images_thresh.append(image_thresh)

        # Encontra componentes
        image_components, n_comp = find_connected_components(image_thresh, image_input_gray)
        error = abs(true_count - n_comp)
        percent_error = (error / true_count) * 100
        total_error += error
        total_percent_error += percent_error
        print(f'[{image_filename}] Encontrados {n_comp}/{true_count} componentes (erro: {percent_error:.2f}%)')
        images_components.append(image_components)
        n_components.append(n_comp)
        print(f'[{image_filename}] Duração: {timeit.default_timer() - start_time:.3f}s')
        cv.imwrite(f'images/contagem_{image_filename}', image_components)
    
    print(f'Erro total em todas as imagens: {total_error}')
    print(f'Erro médio por imagem: {total_error/len(IMAGE_FILENAMES):.2f}')
    print(f'Erro percentual médio: {total_percent_error/len(IMAGE_FILENAMES):.2f}%')
    
    if SHOW_IMAGES:
        plot_images(images_input, images_thresh, images_components, n_components)


if __name__ == '__main__':
    main()
