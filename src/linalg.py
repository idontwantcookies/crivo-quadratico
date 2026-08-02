from collections.abc import Iterable
from numbers import Number
from typing import Callable

import sympy


Vector = list[Number]
Matrix = list[Vector]

def vectorize(f: Callable[[Number], Number]) -> Callable[[Iterable | Number], Iterable | Number]:
    def g(u: Iterable | Number) -> Iterable | Number:
        if isinstance(u, Iterable):
            acc = []
            for x in u:
                acc.append(g(x))
            return acc
        return f(u)
    return g

def vector_mod(v: Vector, p: int) -> Vector:
    u = []
    for x in v:
        u.append(x % p)
    return u

def matrix_mod(A: Matrix, p: int):
    A = A.copy()
    for i, row in enumerate(A):
        A[i] = vector_mod(row, p)
    return A

def swap(A: list, i: int, j: int):
    A[i], A[j] = A[j], A[i]

def find_pivot(A: Matrix, j: int) -> int:
    i, n, m = j, len(A), len(A[0])
    while i < n and i < m and A[i][j] == 0:
        i += 1
    if i in (m, n): return -1
    return i

def sum_vectors(*vectors: list[Vector]) -> Vector:
    acc = []
    for row in zip(*vectors):
        acc.append(sum(row))
    return acc

def scale_vector(u: Vector, alpha: Number) -> Vector:
    w = []
    for x in u:
        w.append(x * alpha)
    return w

def naive_vector_prod(u: Vector, v: Vector) -> Vector:
    w = []
    for x, y in zip(u, v):
        w.append(x * y)
    return w

def transpose(A: Matrix) -> Matrix:
    N = len(A)
    M = len(A[0])
    T = [[0 for _ in range(N)] for _ in range(M)]
    for i in range(N):
        for j in range(M):
            T[j][i] = A[i][j]
    return T

def matrix_prod(A: Matrix, B: Matrix):
    prod = []
    B = transpose(B)
    N, M = len(A), len(B)
    for i in range(N):
        prod.append([])
        for j in range(M):
            prod[i].append(sum(naive_vector_prod(A[i], B[j])))
    return prod

def gauss_reduce_row(row: Vector, pivot: Vector, i: int):
    m = - 1 / pivot[i] * row[i]
    row = sum_vectors(row, scale_vector(pivot, m))
    return row

def rref(A: Matrix) -> Matrix:
    '''Reduz uma matriz N x M à sua forma escalonada (Reduced Row Echelon Form) usando o método
    de Gauss. Complexidade: O(N²)'''
    A = A.copy()
    N = len(A)
    for i in range(N):
        p = find_pivot(A, i)
        if p == -1 or A[p][p] == 0: continue
        swap(A, i, p)
        pivot_row = A[i]
        for j in range(i + 1, N):
            A[j] = gauss_reduce_row(A[j], pivot_row, i)
    return A

def kernel(A: sympy.Matrix | Matrix) -> Matrix:
    '''Base do núcleo de A sobre os racionais. NÃO serve para o crivo
    quadrático, que precisa do núcleo sobre GF(2): use `kernel_gf2`.'''
    A = sympy.Matrix(A)
    ker: list[sympy.Matrix] = A.nullspace()
    return [list(u.transpose()) for u in ker]

def _rows_to_bitmasks(A: Matrix, cols: int) -> list[int]:
    '''Codifica cada linha de A como um inteiro, onde o bit j (peso 2^j)
    vale a entrada da coluna j reduzida mod 2.'''
    masks = []
    for row in A:
        acc = 0
        for j, x in enumerate(row):
            if x % 2:
                acc |= 1 << j
        masks.append(acc)
    return masks

def rref_gf2(A: Matrix) -> tuple[list[int], dict[int, int], int]:
    '''Reduz A à forma escalonada reduzida sobre GF(2) por eliminação
    gaussiana com máscaras de bits (XOR no lugar de somas).

    Retorna (linhas, pivos, cols), onde `linhas` são as máscaras já reduzidas,
    `pivos` mapeia coluna-pivô -> índice da linha correspondente, e `cols` é o
    número de colunas. O posto é len(pivos).

    Cada operação de linha é um único XOR entre inteiros do Python, o que torna
    a eliminação viável nas matrizes de centenas/milhares de colunas geradas
    pelo crivo. Complexidade: O(linhas * colunas) operações de palavra.'''
    if not A: return [], {}, 0
    rows, cols = len(A), len(A[0])
    R = _rows_to_bitmasks(A, cols)
    pivots: dict[int, int] = {}
    r = 0
    for j in range(cols):
        bit = 1 << j
        p = next((i for i in range(r, rows) if R[i] & bit), None)
        if p is None: continue
        R[r], R[p] = R[p], R[r]
        for i in range(rows):
            if i != r and R[i] & bit:
                R[i] ^= R[r]
        pivots[j] = r
        r += 1
        if r == rows: break
    return R, pivots, cols

def _bits_to_vector(h: int, cols: int) -> Vector:
    '''Converte a máscara h num vetor 0/1 de tamanho `cols`, em O(cols).'''
    s = bin(h)[2:][::-1]                       # s[k] é o bit k de h
    v = [0] * cols
    for k, c in enumerate(s):
        if c == '1': v[k] = 1
    return v

def kernel_gf2(A: Matrix) -> Matrix:
    '''Retorna uma base do núcleo de A sobre GF(2): a lista de vetores v com
    entradas em {0, 1} tais que A @ v ≡ 0 (mod 2).

    Cada COLUNA de A vira uma máscara de bits sobre as linhas, e as colunas são
    processadas uma a uma carregando um "histórico" das colunas já combinadas.
    Uma coluna que zera durante a redução dá, pelo seu histórico, exatamente um
    vetor do núcleo. Como cada histórico tem um bit mais alto distinto, os
    vetores obtidos são linearmente independentes e formam uma base.

    O pivô é escolhido por `int.bit_length()`, que é O(1) — daí o ganho sobre
    varrer as linhas testando `linha & (1 << j)`, operação que aloca um inteiro
    proporcional a j a cada teste e domina o custo em matrizes de milhares de
    colunas.

    Exemplo: kernel_gf2([[1, 1, 0], [0, 1, 1]]) => [[1, 1, 1]]'''
    if not A or not A[0]: return []
    cols = len(A[0])
    colunas = [0] * cols
    for i, linha in enumerate(A):
        bit = 1 << i
        for j, x in enumerate(linha):
            if x % 2: colunas[j] |= bit
    pivos: dict[int, tuple[int, int]] = {}     # bit mais alto -> (vetor, histórico)
    basis = []
    for j in range(cols):
        v, h = colunas[j], 1 << j
        while v:
            b = v.bit_length() - 1
            if b not in pivos:
                pivos[b] = (v, h)
                break
            pv, ph = pivos[b]
            v ^= pv
            h ^= ph
        else:
            basis.append(_bits_to_vector(h, cols))
    return basis
