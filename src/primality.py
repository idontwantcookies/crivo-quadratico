from random import randint
from collections import defaultdict

from src.base import gcd, ilog10, oddify
from src.modular_arithmetic import powmod


def miller_test(n:int, b:int, k: int, q: int):
    '''Usa o teste de Miller para testar se um número é primo.
    Caso n  seja primo, phi(n) = n - 1, portanto espera-se que
    b^(n - 1) = 1 mod n, pelo pequeno teorema de Fermat. 
    Quando essa igualdade não é atendida, sabemos que n é composto.
    Como fatorar n - 1 é custoso, o algoritmo itera sobre alguns
    dos divisores de n - 1: d = q, 2q, 4q, 8q, ..., n - 1. Caso qualquer
    um deles seja tal que b^d = 1 mod n, então sabemos que
    b^(n-1) = 1 mod n, portanto n pode ser primo e o teste é
    inconclusivo. Caso contrário, se chegamos a b^(n-1) != 1, então
    n certamente é composto.
    O teste de Miller retorna True caso o número seja POSSIVELMENTE
    primo, e False caso seja CERTAMENTE composto.
    Complexidade: O(k * log(n)) = O(log²(n)).
    '''
    if n == 2 or n == -2: return True
    if n % 2 == 0: return False
    # Um gcd(n, b) que seja fator PRÓPRIO de n prova que n é composto; não é
    # evidência de que n é primo. O sinal estava invertido aqui, o que inflava
    # bastante a taxa de falsos positivos de prime_miller_rabin sempre que a
    # base aleatória calhava de dividir n (ex.: n=21 era classificado como
    # primo em ~1 a cada 1400 chamadas).
    # O caso d == n (b múltiplo de n, fora do intervalo que prime_miller_rabin
    # sorteia) é diferente: a base não carrega informação nenhuma, então o
    # honesto é devolver "inconclusivo" em vez de acusar n de composto.
    d = gcd(n, b)
    if d == n: return True
    if d != 1: return False
    r = powmod(b, q, n)
    if r == 1 or r == n - 1: return True
    for _ in range(k):
        r = powmod(r, 2, n)
        if r == n - 1: return True
    return False

def prime_miller_rabin(n:int, primes:list[int]=None, rep:int=None):
    '''Executa `rep` iterações com bases aleatórias do teste de Miller 
    para checar se n é primo. Returna True se o número provavelmente é primo,
    e False caso seja certamente composto.
    Complexidade de tempo: O(rep * log²(n))'''
    n = abs(n)
    primes = primes or [2]
    if n < 2: return False
    if n == 2: return True
    if n in primes: return True
    if n < primes[-1]: return False
    rep = rep or max(10, ilog10(n) + 1)
    k, q = oddify(n - 1)
    for _ in range(rep):
        b = randint(2, n - 1)
        if not miller_test(n, b, k, q): return False
    return True

def eratosthenes_sieve(n: int) -> list[int]:
    '''
    Usa o crivo de Eratóstenes para retornar uma lista com 
    todos os primos no intervalo fechado [0,n].
    Complexidade de tempo: O(n * log(log(n)).
    '''
    L = defaultdict(lambda: True)
    L[0], L[1] = False, False
    primes = []
    for i in range(n + 1):
        if L[i]:
            primes.append(i)
            for j in range(i**2, n + 1, i):
                L[j] = False
    return primes
