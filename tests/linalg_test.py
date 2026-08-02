import random
from fractions import Fraction

import pytest
import sympy

from src import linalg

def test_vectorize():
    mod5 = linalg.vectorize(lambda x: x % 5)
    assert mod5([4, 8, 2, 7, 11, 23]) == [4, 3, 2, 2, 1, 3]
    assert mod5([[11, 17], [-4, 10]]) == [[1, 2], [1, 0]]

def test_vector_mod():
    v = [6, 1, 4, 2]
    linalg.vector_mod(v, 3)
    assert linalg.vector_mod(v, 3) == [0, 1, 1, 2]

def test_add_vectors():
    u = [10, 5, 7, 13, 19]
    v = [-3, -9, 4, 0, 12]
    assert linalg.sum_vectors(u, v) == [7, -4, 11, 13, 31]

def test_find_pivot():
    A = [[0, 3, 7],
         [0, 0, 1],
         [1, 0, 3],
         [0, 1, 7]]
    assert linalg.find_pivot(A, 0) == 2
    # A[3][1] == 1, então o pivô da coluna 1 está na linha 3. Este caso
    # esperava -1 porque find_pivot parava a busca no número de colunas (3).
    assert linalg.find_pivot(A, 1) == 3
    assert linalg.find_pivot(A, 2) == 2

def test_find_pivot2():
    A = [[0, 0, 3, 0],
         [0, 0, 0, 1],
         [0, 1, 0, 2]]
    assert linalg.find_pivot(A, 1) == 2
    assert linalg.find_pivot(A, 3) == -1     # não há linha 3 numa matriz 3x4

def test_find_pivot_coluna_nula():
    A = [[0, 1],
         [0, 2],
         [0, 3]]
    assert linalg.find_pivot(A, 0) == -1

def test_find_pivot_mais_linhas_que_colunas():
    assert linalg.find_pivot([[0, 0], [0, 0], [1, 1]], 0) == 2

def test_find_pivot_coluna_inexistente():
    assert linalg.find_pivot([[1, 2], [3, 4]], 5) == -1
    assert linalg.find_pivot([], 0) == -1

def test_find_pivot_desde():
    A = [[1, 0],
         [1, 0],
         [1, 0]]
    assert linalg.find_pivot(A, 0) == 0
    assert linalg.find_pivot(A, 0, desde=1) == 1
    assert linalg.find_pivot(A, 0, desde=3) == -1

def test_matrix_mod():
    A = [[2, 10],
         [4, 13]]
    assert linalg.matrix_mod(A, 7) == [[2, 3],
                                       [4, 6]]

def test_swap():
    v = [1, 2, 3, 4, 5]
    linalg.swap(v, 0, 3)
    assert v == [4, 2, 3, 1, 5]

def test_transpose():
    A = [[0, 3, 7, 4],
         [0, 0, 1, 2],
         [1, 0, 3, 11]]
    assert linalg.transpose(A) == [[0, 0, 1],
                                   [3, 0, 0],
                                   [7, 1, 3],
                                   [4, 2, 11]]

def test_matrix_prod():
    A = [[1, 2, 3],
         [4, 5, 6]]
    B = [[1, 2],
         [3, 4],
         [5, 6]]
    assert linalg.matrix_prod(A, B) == [[22, 28],
                                        [49, 64]]

def test_naive_vector_prod():
    u = [1, 2, 3]
    v = [9, 8, 7]
    assert linalg.naive_vector_prod(u, v) == [9, 16, 21]


@pytest.mark.parametrize('A,A_reduced', [
    [[[5, 2, 3],
      [2, 4, 1],
      [1, 0, 1]], [[5, 2, 3],
                   [0.0, 3.2, -0.20000000000000018],
                   [0.0, 0.0, 0.3749999999999999]]],
    [[[5, 2],
      [2, 4],
      [1, 0]], [[5, 2],
                [0.0, 3.2],
                [0.0, 0.0]]],
    [[[5, 2, 3],
      [2, 4, 1]], [[5, 2, 3],
                   [0.0, 3.2, -0.20000000000000018]]]
])
def test_rref(A, A_reduced):
    assert linalg.rref(A) == A_reduced

def test_rref_nao_altera_a_entrada():
    A = [[5, 2, 3],
         [2, 4, 1],
         [1, 0, 1]]
    original = [row[:] for row in A]
    linalg.rref(A)
    assert A == original

def test_rref_com_coluna_sem_pivo():
    '''Coluna nula no meio: a coluna avança, mas a linha do pivô não. Antes o
    mesmo índice servia para linha e coluna e o resultado saía errado.'''
    A = [[0, 2, 1],
         [0, 4, 3],
         [0, 6, 8]]
    R = linalg.rref(A)
    assert [linha[0] for linha in R] == [0, 0, 0]
    assert R[0][1] != 0                      # pivô da coluna 1 na linha 0
    assert R[1][1] == 0 and R[2][1] == 0     # zerado abaixo do pivô
    # A coluna 0 não gastou linha: o pivô da coluna 2 cai na linha 1, não na 2.
    assert R[1][2] != 0
    assert R[2] == [0, 0, 0]                 # posto 2: a última linha zera

def test_rref_propriedades_em_matrizes_aleatorias():
    '''Compara `rref` com um oráculo independente (sympy) em 200 matrizes
    aleatórias. Usa Fraction para a aritmética ficar exata: com float, o posto
    de uma matriz empilhada não é confiável.'''
    random.seed(5)
    for _ in range(200):
        N, M = random.randint(1, 6), random.randint(1, 6)
        densidade = random.choice([0.3, 0.6, 1.0])
        A = [[Fraction(random.randint(-5, 5)) if random.random() < densidade else Fraction(0)
              for _ in range(M)] for _ in range(N)]
        R = linalg.rref(A)

        pivos = []
        for linha in R:
            nao_nulos = [j for j, x in enumerate(linha) if x != 0]
            pivos.append(nao_nulos[0] if nao_nulos else None)
        ocupados = [p for p in pivos if p is not None]

        # forma escalonada: pivôs estritamente crescentes, linhas nulas no fim
        assert ocupados == sorted(ocupados) and len(set(ocupados)) == len(ocupados)
        assert all(p is None for p in pivos[len(ocupados):])
        # posto e espaço-linha preservados
        posto = sympy.Matrix(A).rank()
        assert len(ocupados) == posto
        assert sympy.Matrix(A + R).rank() == posto


def test_rref_preserva_o_posto():
    '''O número de linhas não-nulas da forma escalonada é o posto da matriz.'''
    casos = [
        ([[1, 2], [2, 4]], 1),               # linhas dependentes
        ([[1, 2], [3, 4]], 2),
        ([[0, 0], [0, 0]], 0),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2),
        ([[1, 0], [0, 1], [1, 1]], 2),       # mais linhas que colunas
    ]
    for A, posto in casos:
        R = linalg.rref(A)
        nao_nulas = sum(1 for linha in R if any(abs(x) > 1e-9 for x in linha))
        assert nao_nulas == posto, f'posto de {A} deveria ser {posto}'
