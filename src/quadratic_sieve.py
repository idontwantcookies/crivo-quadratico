from math import exp, sqrt, log, ceil
from itertools import product
from collections import OrderedDict, defaultdict
from random import randrange
from time import time

from src.base import isqrt, gcd
from src.factorization import factor_with_limited_primes
from src.linalg import (Matrix, Vector, transpose, kernel_gf2, sum_vectors, scale_vector,
                        vector_mod, matrix_mod)
from src.modular_arithmetic import find_non_square, is_square, msqrt
from src.primality import eratosthenes_sieve
from src.util import Powers, SieveTimeout


def find_B(n: int) -> int:
    '''Retorna o limite B do crivo quadrático, onde B é o tamanho máximo de um primo.
    Fonte: https://risencrypto.github.io/QuadraticSieve/'''
    return ceil(exp(sqrt(log(n) * log(log(n))))**(1/sqrt(2))) + 1

def euler_sieve_method(n: int, primes: list[int]) -> list[int]:
    '''Criva os primos de acordo com o critério de Euler; ou seja, filtra a lista de primos
    para deixar apenas aqueles fazem n ser quadrado mod p.'''
    return list(filter(lambda p: is_square(n, p), primes))

def setup(n: int):
    B = find_B(n)
    primes = eratosthenes_sieve(B)
    primes = euler_sieve_method(n, primes)
    primes.insert(0, -1)
    M = len(primes) + 5
    return B, M, primes

def quadratic_sieve_aux(n: int, xj: int, S: dict[int, Powers], primes: list[int]):
    '''Função auxiliar para o crivo quadrático. Ela recebe um número grande n que se deseja fatorar,
    um inteiro qualquer xj, e decompõe xj²-n em potências de primos presentes em `primes`,
    tal que
    primes = [p1, p2, ..., pk]
    xj²-n ≡ p1^alpha1 * p2^alpha2 * ... * pk^alphak.
    Se xj²-n for B-smooth, ou seja, pode ser perfeitamente decomposto em primos na lista finita `primes`,
     então a decomposição é adicionada ao dicionário S.

    Testa UM candidato por divisão por tentativa sobre toda a base de fatores.
    É o método de Dixon; `sieve_block` faz o mesmo trabalho para um intervalo
    inteiro a um custo muito menor. Mantida por ser usada nos testes.'''
    if xj in S.keys(): return
    decomp, u = factor_with_limited_primes(xj * xj % n, primes)
    if u == 1:
        # Number is B-smooth
        S[xj] = decomp

def sieve_roots(n: int, primes: list[int]) -> dict[int, list[int]]:
    '''Para cada primo p da base de fatores, resolve x² ≡ n (mod p) e devolve
    as raízes. Como a base já foi filtrada pelo critério de Euler, n é resíduo
    quadrático mod p e as raízes existem.

    É isto que transforma Dixon em crivo: em vez de testar todo candidato
    contra todo primo, sabemos de antemão que p divide Q(x) = x² - n
    exatamente nas posições x ≡ r (mod p).'''
    roots: dict[int, list[int]] = {}
    for p in primes:
        if p < 2: continue                      # a coluna do -1 não é crivada
        if p == 2:
            # x² ≡ n (mod 2) tem raiz única: x ≡ n (mod 2)
            roots[p] = [n % 2]
            continue
        if n % p == 0:
            roots[p] = [0]
            continue
        r = msqrt(n % p, p, find_non_square(p))
        roots[p] = [r] if r == p - r else sorted({r, p - r})
    return roots

def sieve_block(n: int, primes: list[int], roots: dict[int, list[int]],
                inicio: int, tamanho: int) -> OrderedDict:
    '''Peneira o intervalo [inicio, inicio + tamanho) e devolve as relações
    B-smooth encontradas: {x: {p: expoente}} com x² ≡ prod(p^expoente) (mod n).

    Usa Q(x) = x² - n (não x² mod n), que é o polinômio cujas raízes módulo p
    conhecemos. Para cada p, só as posições x ≡ r (mod p) são tocadas, e delas
    o fator p é dividido de fato — nada de limiares de logaritmo, então não há
    falso positivo nem falso negativo. O custo cai de
    O(tamanho * |base|) para O(tamanho * sum(1/p)) ≈ O(tamanho * ln ln B).'''
    restos = [(inicio + i) * (inicio + i) - n for i in range(tamanho)]
    expoentes: list[defaultdict] = [defaultdict(lambda: 0) for _ in range(tamanho)]
    for p in primes:
        if p < 2: continue
        for r in roots.get(p, ()):
            # primeiro índice i >= 0 com (inicio + i) ≡ r (mod p)
            for i in range((r - inicio) % p, tamanho, p):
                if restos[i] == 0: continue
                while restos[i] % p == 0:
                    restos[i] //= p
                    expoentes[i][p] += 1
    achados = OrderedDict()
    for i, resto in enumerate(restos):
        if resto == 1:
            achados[inicio + i] = expoentes[i]
    return achados

def build_matrix_of_powers(multiplicities: dict[int, Powers], primes: list[int]) -> Matrix:
    A = []
    for _xj, powers in multiplicities.items():
        row = []
        for p in primes:
            row.append(powers[p])
        A.append(row)
    return transpose(A)

def kernel_solutions(ker: list[Vector]) -> list[Vector]:
    '''Gera TODAS as 2^dim combinações lineares da base `ker` sobre GF(2).

    Só é utilizável para dimensões pequenas: no crivo quadrático dim cresce
    rápido (25 já em 15 dígitos, ou seja 33 milhões de vetores). Para o laço
    de fatoração use `iter_kernel_solutions`, que não materializa o espaço.'''
    solutions = []
    dim = len(ker)
    if dim == 0: return solutions
    n = len(ker[0])
    S = product([0, 1], repeat=dim)
    for comb in S:
        out = [0] * n
        for alpha, u in zip(comb, ker):
            u = scale_vector(u, alpha)
            out = sum_vectors(u, out)
            out = vector_mod(out, 2)
        solutions.append(out)
    return solutions

def iter_kernel_solutions(ker: list[Vector], limit: int = 512):
    '''Percorre soluções não-nulas do núcleo sem materializar as 2^dim
    combinações. Primeiro os próprios vetores da base, depois combinações
    aleatórias distintas, até `limit` soluções.

    Cada solução tem probabilidade ~1/2 de produzir um fator não-trivial, então
    poucas dezenas de tentativas bastam na prática; enumerar o espaço inteiro é
    desnecessário e inviável.'''
    dim = len(ker)
    if dim == 0: return
    n = len(ker[0])
    seen = set()

    def emit(v: Vector):
        t = tuple(v)
        if not any(t) or t in seen: return None
        seen.add(t)
        return v

    for u in ker:
        v = emit(vector_mod(u, 2))
        if v is not None: yield v

    tentativas = 0
    while len(seen) < limit and tentativas < 8 * limit:
        tentativas += 1
        mask = randrange(1, 1 << dim)
        out = [0] * n
        for i, u in enumerate(ker):
            if (mask >> i) & 1:
                out = sum_vectors(out, u)
        v = emit(vector_mod(out, 2))
        if v is not None: yield v

def join_powers(*decompositions: list[Powers]) -> Powers:
    '''Multiplica números decompostos em primos, somando seus expoentes quando ocorre
    colisão nas chaves (primos) dos respectivos dicionários.'''
    acc = defaultdict(lambda: 0)
    for decomp in decompositions:
        for p, alpha in decomp.items():
            acc[p] += alpha
    return acc

def isqrt_powers(decomp: Powers) -> Powers:
    decomp = decomp.copy()
    for p, alpha in decomp.items():
        if alpha % 2 != 0: raise ValueError("Given number is not a perfect square.")
        decomp[p] = alpha // 2
    return decomp

def compose(decomp: Powers) -> int:
    acc = 1
    for p, alpha in decomp.items():
        acc *= p**alpha
    return acc

def compose_from_solution(S: OrderedDict[int, Powers], solution: list[int]) -> int:
    prod = 1
    decomp = defaultdict(lambda: 0)
    for i, (guess, powers) in enumerate(S.items()):
        if solution[i] == 1:
            prod *= guess
            decomp = join_powers(decomp, powers)
    decomp = isqrt_powers(decomp)
    return prod, compose(decomp)


# Pré-passagem de divisão por tentativa pelos primos até B. É O(|base|), não
# O(sqrt(n)), e resolve os n pequenos cujo menor fator já está na base — casos
# em que o crivo mal tem intervalo para trabalhar. Os testes a desligam para
# exercitar o crivo de verdade.
TRIAL_DIVISION = True

MAX_TRIAL_DIVISION = 10**6

def small_factor(n: int, limite: int) -> int | None:
    '''Procura um fator primo de n até `limite` por divisão por tentativa.
    Devolve None se não houver.

    Não dá para usar a base de fatores aqui: o critério de Euler descarta
    justamente os p que dividem n (para esses, n não é resíduo quadrático
    mod p). Então a busca é feita sobre todos os inteiros até o limite.'''
    limite = min(limite, isqrt(n), MAX_TRIAL_DIVISION)
    if n % 2 == 0 and n != 2: return 2
    p = 3
    while p <= limite:
        if n % p == 0: return p
        p += 2
    return None

def collect_relations(n: int, primes: list[int], alvo: int,
                      start: float, timeout: float) -> OrderedDict:
    '''Peneira blocos sucessivos a partir de isqrt(n)+1 até juntar `alvo`
    relações B-smooth. O bloco cresce quando o rendimento é baixo, para que n
    grandes (onde as relações são raras) não paguem o custo fixo por bloco.'''
    roots = sieve_roots(n, primes)
    S: OrderedDict[int, Powers] = OrderedDict()
    x0 = isqrt(n) + 1                 # isqrt, não sqrt: float erra acima de 2^53
    x = x0
    bloco = max(1024, min(1 << 16, 4 * len(primes)))
    # Q(x) cresce com x, então a partir de certo ponto não há mais relações a
    # colher e insistir só queima o tempo limite. Vale para n pequeno demais
    # para o crivo, onde a base de fatores tem meia dúzia de primos.
    alcance = max(1 << 20, 256 * len(primes) * len(primes))
    while len(S) < alvo:
        if time() - start > timeout:
            raise SieveTimeout(
                f'Tempo limite de {timeout}s excedido: {len(S)} de {alvo} relações.')
        if x - x0 > alcance:
            raise RuntimeError(
                f'Não foi possível construir um sistema de equações para n: '
                f'{len(S)} de {alvo} relações em {alcance} candidatos.')
        achados = sieve_block(n, primes, roots, x, bloco)
        S.update(achados)
        x += bloco
        if len(achados) * 4 < bloco // 64:
            bloco = min(bloco * 2, 1 << 20)
    return S

def quadratic_sieve(n: int, timeout: float = 15) -> int:
    '''Implementação do crivo quadrático baseada em Collier:
    https://www.dcc.ufrj.br/~collier/CursosGrad/topicos/CrivoQuadratico.html

    Devolve um fator não-trivial de n. Levanta `SieveTimeout` se estourar
    `timeout` segundos, e `RuntimeError` se não achar fator não-trivial.'''
    start = time()
    if n % 2 == 0 and n != 2: return 2
    raiz = isqrt(n)
    if raiz * raiz == n: return raiz          # quadrado perfeito: a ≡ ±b sempre
    B, M, primes = setup(n)
    if TRIAL_DIVISION:
        d = small_factor(n, B)
        if d is not None: return d
    S = collect_relations(n, primes, M + 1, start, timeout)
    # É preciso ter mais relações do que primos na base de fatores para
    # garantir que o sistema sobre GF(2) tenha solução não-trivial.
    if len(S) <= len(primes):
        raise RuntimeError('Não foi possível construir um sistema de equações para n.')
    A = build_matrix_of_powers(S, primes)
    A = matrix_mod(A, 2)
    # O núcleo tem de ser calculado sobre GF(2): sobre os racionais ele perde
    # relações e produz vetores com entradas fracionárias.
    ker = kernel_gf2(A)
    for sol in iter_kernel_solutions(ker):
        a, b = compose_from_solution(S, sol)
        a, b = a % n, b % n
        assert (a * a - b * b) % n == 0
        d = abs(gcd(a - b, n))
        if d not in (1, n):
            return d
        if time() - start > timeout:
            raise SieveTimeout(f'Tempo limite de {timeout}s excedido na busca de fatores.')
    raise RuntimeError('Não foi possível encontrar um fator não-trivial para n.')
