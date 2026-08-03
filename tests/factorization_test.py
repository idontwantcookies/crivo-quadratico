import pytest


from src import factorization

@pytest.mark.parametrize("x,factors,phi", [
    [48, {2:4, 3:1}, 16],
    [11, {11:1}, 10]
])
def test_totient(x, factors, phi):
    assert factorization.totient(x, factors) == phi

@pytest.mark.parametrize('n', [12, 850903, 717967279050961])
def test_pollard_rho(n):
    x = factorization.pollard_rho_factor(n)
    y = n // x
    assert x * y == n

@pytest.mark.parametrize("n,primes,f", [
    [12, None, {2: 2, 3: 1}],
    [717967279050961, None, {12657973: 1, 56720557: 1}],
    [100, None, {2: 2, 5: 2}],
    [15, [2, 3, 5], {3: 1, 5: 1}]
])
def test_factors(n, primes, f):
    assert factorization.pollard_rho_prime_power_decomposition(n, primes) == f

@pytest.mark.parametrize("n,p,u,alpha", [
    [51, 3, 17, 1],
    [16, 2, 1, 4],
    [16, 3, 16, 0]
])
def test_factor_out(n, p, u, alpha):
    assert factorization.factor_out(n, p) == (u, alpha)

@pytest.mark.parametrize("n,primes,powers,u", [
    [2**3 * 3**2 * 11**3 * 13 * 17, [-1, 2, 3, 5, 7, 11], {-1:0, 2:3, 3:2, 5:0, 7:0, 11:3}, 13*17],
    [-22, [-1, 2, 3, 5], {-1:1, 2:1, 3:0, 5:0}, 11]
])
def test_factor_with_limited_primes(n, primes, powers, u):
    assert factorization.factor_with_limited_primes(n, primes) == (powers, u)


# --------------------------------------------------------------------------
# Regressão: a multiplicidade propagada na recursão de
# pollard_rho_prime_power_decomposition usava `count + i - 1` em vez de
# `count * i`.
#
# O parâmetro `count` significa "esta chamada fatora n**count". Como
# factor_out garante n = x**i * y, segue que n**count = x**(i*count) * y**count,
# logo o ramo de x tem de receber i*count. Soma e produto só coincidem quando
# count == 1 ou i == 1, então o bug exigia um fator repetido DENTRO de outro
# fator repetido para aparecer — daí ter passado despercebido.
#
# Como o caminho da recursão depende do fator que o Pollard rho sorteia, cada
# caso roda REPS vezes: alguns n só erram em parte dos caminhos.
# --------------------------------------------------------------------------

REPS = 25

@pytest.mark.parametrize("n,f", [
    [3**6, {3: 6}],                                     # errava em 100% dos caminhos
    [2**6, {2: 6}],
    [2**8, {2: 8}],
    [2**12, {2: 12}],
    [5**4, {5: 4}],
    [2**6 * 5, {2: 6, 5: 1}],
    [2**6 * 3**2, {2: 6, 3: 2}],
    [2**4 * 3**4, {2: 4, 3: 4}],
    [2**10 * 3**5 * 7, {2: 10, 3: 5, 7: 1}],
    # phi(8763841): o caso concreto que quebrava o logaritmo discreto.
    [8763840, {2: 6, 3: 2, 5: 1, 17: 1, 179: 1}],
])
def test_prime_power_decomposition_fatores_repetidos_aninhados(n, f):
    for _ in range(REPS):
        assert factorization.pollard_rho_prime_power_decomposition(n) == f


@pytest.mark.parametrize("n", [
    64, 128, 256, 729, 625, 1024, 4096, 2**6 * 3**2, 2**4 * 3**4,
    1741824, 8763840, 2**7 * 3**3 * 5**2, 11**4, 13**3 * 2**5,
])
def test_prime_power_decomposition_reconstroi_n(n):
    '''Invariante que qualquer fatoração tem de satisfazer: multiplicar os
    fatores de volta devolve n. É o teste que mais barato pegaria o bug, já
    que ele produzia dicionários de aparência normal com o produto errado
    (8763840 saía como 4381920, exatamente a metade).'''
    for _ in range(REPS):
        f = factorization.pollard_rho_prime_power_decomposition(n)
        produto = 1
        for p, e in f.items():
            produto *= p**e
        assert produto == n, f"fatoração {dict(f)} multiplica para {produto}, não {n}"
