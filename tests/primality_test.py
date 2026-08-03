import pytest

from src import primality
from src.base import gcd, oddify
from .big_numbers import primes


@pytest.mark.parametrize("n,P", [
    [10, [2, 3, 5, 7]],
    [30, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]],
    [5, [2, 3, 5]]
])
def test_erastosthenes_sieve(n, P):
    assert primality.eratosthenes_sieve(n) == P


@pytest.mark.parametrize("p,result", primes)
def test_primes_miller_rabin(p, result):
    result = bool(result)
    assert primality.prime_miller_rabin(p) == result

@pytest.mark.parametrize("n,result", [
    [2, True],
    [3, True],
    [4, False],
    [10, False],
    [45, False],
    [101, True],
    [211, True],
    [21, False]
])
def test_miller_rabin_small(n, result):
    assert primality.prime_miller_rabin(n) == result


# --------------------------------------------------------------------------
# Regressão: o sinal de miller_test na checagem do gcd estava invertido.
#
# Quando gcd(n, b) é um fator PRÓPRIO de n, isso é uma prova construtiva de
# que n é composto. A versão antiga devolvia True ("talvez primo") nesse caso,
# transformando a evidência mais forte possível de composição em evidência de
# primalidade. Efeito prático: prime_miller_rabin(21) devolvia True em cerca
# de 1 a cada 1400 chamadas, e a fatoração acima parava cedo demais.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n,b", [
    [21, 3],        # gcd = 3
    [21, 7],        # gcd = 7
    [21, 14],       # gcd = 7, base não trivial
    [15, 5],
    [15, 12],       # gcd = 3
    [45, 9],        # gcd = 9, ele próprio composto
    [561, 33],      # Carmichael: passa no teste de Fermat, mas 33 | 561
    [1105, 65],     # Carmichael
])
def test_miller_test_base_com_fator_comum_prova_composicao(n, b):
    '''Se a base compartilha um fator próprio com n, o teste tem de acusar
    composição (False), nunca devolver "inconclusivo".'''
    k, q = oddify(n - 1)
    assert gcd(n, b) not in (1, n), "caso de teste mal construído"
    assert primality.miller_test(n, b, k, q) is False


@pytest.mark.parametrize("n", [9, 15, 21, 25, 27, 33, 35, 45, 49, 561, 1105])
def test_miller_test_todas_as_bases_nao_coprimas_reprovam(n):
    '''Varredura exaustiva: para um composto n, TODA base em [2, n-1] que não
    seja coprima com n deve reprovar. Não depende de sorte no sorteio.'''
    k, q = oddify(n - 1)
    for b in range(2, n):
        if gcd(n, b) != 1:
            assert primality.miller_test(n, b, k, q) is False, f"base b={b} não reprovou n={n}"


@pytest.mark.parametrize("p,b", [
    [7, 7],         # b == p
    [7, 14],        # b múltiplo de p
    [11, 11],
    [13, 26],
    [101, 303],
])
def test_miller_test_base_multipla_de_primo_e_inconclusiva(p, b):
    '''Contrapartida do teste acima: quando gcd(n, b) == n a base não carrega
    informação (b ≡ 0 mod n), então o resultado honesto é "inconclusivo".
    Acusar composição aqui transformaria um primo em falso negativo.'''
    k, q = oddify(p - 1)
    assert gcd(p, b) == p, "caso de teste mal construído"
    assert primality.miller_test(p, b, k, q) is True


@pytest.mark.parametrize("n", [9, 15, 21, 25, 27, 33, 35, 45, 49])
def test_prime_miller_rabin_sem_falso_positivo(n):
    '''prime_miller_rabin é probabilístico, mas para estes compostos pequenos
    a probabilidade de falso positivo é desprezível (não têm "strong liars"
    além de b = n-1). Com o bug do gcd, n=21 falhava ~1 em 1400, então mil
    repetições detectam a regressão com folga.'''
    assert not any(primality.prime_miller_rabin(n) for _ in range(1000))
