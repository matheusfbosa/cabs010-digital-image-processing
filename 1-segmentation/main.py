#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE = 'arroz.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 5
LARGURA_MIN = 5
N_PIXELS_MIN = 100
QUATRO_VIZINHOS_HABILITADO = True
QUATRO_VIZINHOS = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
OITO_VIZINHOS = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]

#===============================================================================

def binariza(img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''

    # TODO: escreva o código desta função.
    # Dica/desafio: usando a função np.where, dá para fazer a binarização muito
    # rapidamente, e com apenas uma linha de código!
    
    # for channel in range(img.shape[2]):
    #     for row in range(img.shape[0]):
    #         for col in range(img.shape[1]):
    #             if img[row, col, channel] > threshold:
    #                 img[row, col, channel] = 1
    #             else:
    #                 img[row, col, channel] = 0
    # return img
    
    return np.where(img > threshold, 1.0, 0.0)

#-------------------------------------------------------------------------------

def flood_fill(img, x, y, label, original_value=1):
    # Se o pixel está fora da imagem, retorna 0 e as coordenadas do pixel.
    if x < 0 or x >= img.shape[0] or y < 0 or y >= img.shape[1]:
        return 0, x, y, x, y

    # Se o pixel é diferente do valor original, retorna 0 e as coordenadas do pixel.
    if img[x, y] != original_value:
        return 0, x, y, x, y

    # Marca o pixel com o novo rótulo.
    img[x, y] = label

    # Inicializa contadores e coordenadas do retângulo envolvente.
    n_pixels = 1
    top, left, bottom, right = x, y, x, y

    # Especifica os vizinhos.
    vizinhos = QUATRO_VIZINHOS if QUATRO_VIZINHOS_HABILITADO else OITO_VIZINHOS

    # Para cada vizinho.
    for dx, dy in vizinhos:
        contador, t, l, b, r = flood_fill(img, x + dx, y + dy, label, original_value)
        n_pixels += contador
        top, left = min(top, t), min(left, l)
        bottom, right = max(bottom, b), max(right, r)

    # Retorna o número de pixels e as coordenadas do retângulo envolvente.
    return n_pixels, top, left, bottom, right

def rotula(img, largura_min, altura_min, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''

    # TODO: escreva esta função.
    # Use a abordagem com flood fill recursivo.

    rotulos = []
    rotulo = 0.1

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # Se encontrou novo componente.
            if img[row, col] == 1:
                n_pixels, top, left, bottom, right = flood_fill(img, row, col, rotulo)

                # Calcula largura e altura do retângulo envolvente.
                largura = right - left + 1
                altura = bottom - top + 1

                # Se atende aos critérios mínimos.
                if largura >= largura_min and altura >= altura_min and n_pixels >= n_pixels_min:
                    # Adiciona o componente à lista de componentes.
                    rotulos.append({
                        'label': rotulo,
                        'n_pixels': n_pixels,
                        'T': top, 'L': left, 'B': bottom, 'R': right
                    })

                    # Atualiza para o próximo rótulo.
                    rotulo += 0.1

    return rotulos

#===============================================================================

def main():
    # Abre a imagem em escala de cinza.
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype(np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza(img, THRESHOLD)
    cv2.imshow('01 - binarizada', img)
    cv2.imwrite('01 - binarizada.png', img*255)

    start_time = timeit.default_timer()
    componentes = rotula(img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len(componentes)
    print('Tempo: %f' % (timeit.default_timer() - start_time))
    print('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle(img_out, (c['L'], c['T']), (c['R'], c['B']), (0,0,1))

    cv2.imshow('02 - out', img_out)
    cv2.imwrite('02 - out.png', img_out*255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

#===============================================================================
