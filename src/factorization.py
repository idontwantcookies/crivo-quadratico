from collections import Counter
from random import randint
from time import time

from src.base import poly, gcd
from src.primality import prime_miller_rabin
from src.util import error, Powers


def totient(x:int, f:Powers) -> int:
    '''
    Calcula o totiente de x, phi, tal que a^phi = 1 mod x para qualquer a.
    É preciso conhecer a fatoração de x.
    Exemplo: totient(40, {2:3,5:1}) => 16
    '''
    phi = x
    for p in f.keys():
        phi = phi * (p - 1) // p
    return phi

def factor_out(n: int, p: int) -> tuple[int, int]:
    '''Retorna u, alpha tais que n = p^alpha * u'''
    u, alpha = n, 0
    while u % p == 0:
        u //= p
        alpha += 1
    return u, alpha

def factor_with_limited_primes(n: int, primes: list[int]) -> tuple[Powers, int]:
    '''Retorna u, {p1: alpha1, p2:alpha2, ..., pk:alphak} tais que
    k é o tamanho da lista de primos passada, e
    n = p1^alpha1 * p2^alpha2 * ... * pk^alphak * u
    onde u é um número cuja fatoração em potências de primos não possui nenhum
    primo na lista `primes`.'''
    if n == 0: raise ValueError("n cannot be zero.")
    u, powers = n, {}
    if -1 in primes:
        powers[-1] = 1 if n < 0 else 0
        primes = primes[1:]
    u = abs(u)
    for p in primes:
        u, alpha = factor_out(u, p)
        powers[p] = alpha
    return powers, u

def pollard_rho_factor(n: int, timeout:int=15) -> int:
    '''Usa o algoritmo Pollard's rho para encontrar um fator de n.
    Retorna o valor encontrado.
    Exemplo: pollard_rho_factor(40) => 8
    Complexidade: O(rep * n) no pior caso.
    Complexidade amortizada: O(sqrt(p)), onde p é o maior fator primo de n.'''
    if prime_miller_rabin(n): raise ValueError(f"Called pollard_rho_factor() on n={n}, but it looks like n is prime.")
    x = 2
    c = [1, 0, 1]			# pseudo-random poly coefficients i.e. 1x²+0x+1
    start = time()
    while time() - start < timeout:
        p = poly(*c)		# pseudo-random polynom
        T, H = x, x			# tortoise and the hare
        for _ in range(n):
            T = p(T) % n
            H = p(p(H)) % n
            d = gcd(T - H, n)
            if 1 < d < n: return d
            if d == n:
                # restart the process with different inputs
                x = randint(0, n - 1)		 				# arbitrary starting value for x
                c = [randint(0, n - 1) for _ in range(3)]	# arbitrary coefficients
                break
    error(f"Tempo excedido: não foi possível encontrar um fator de n - 1. Tempo máximo: {timeout}")

def pollard_rho_prime_power_decomposition(n: int, primes:list[int]=None, count=1) -> Counter[int]:
    '''
    Usa o algoritmo Pollard's rho para encontrar a decomposição em potências de
    primos de n.
    Complexidade: O(r * sqrt(p)), onde r é o número de fatores primos de n, e p
    é o maior fator primo de n.
    Exemplo: pollard_rho_prime_power_decomposition(40) => {2: 3, 5: 1}
    '''
    if n == 1: return Counter()
    primes = primes or []
    if prime_miller_rabin(n): return Counter({n: count})
    for p in primes:
        if n % p == 0:
            x = p
            break
    else:
        x = pollard_rho_factor(n)
    y, i = factor_out(n, x)
    x_factors = pollard_rho_prime_power_decomposition(x, primes, count + i - 1)
    y_factors = pollard_rho_prime_power_decomposition(y, primes, count)
    return x_factors + y_factors
