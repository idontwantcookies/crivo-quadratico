def ilog10(n:int) -> int:
    '''Retorna o logaritmo inteiro de n na base 10. Complexidade: O(log(n))
    Ex.: ilog10(1031) => 3'''
    x = 0
    while n >= 10:
        n //= 10
        x += 1
    return x

def isqrt(n: int) -> int:
    '''Retorna a raiz quadrada inteira x de n, x² <= n, usando
    busca binária. Complexidade: O(log(n)).
    Exemplo: isqrt(51) => 7'''
    L = 0
    R = n + 1
    while L != R - 1:
        M = (L + R) // 2
        if M * M <= n:
            L = M
        else:
            R = M
    return L

def gcd(a:int, b:int) -> int:
    '''Implementa recursivamente o cálculo do MDC entre a e b
    usando o algoritmo de Euclides. Complexidade: O(log(min(a, b))).
    Exemplo: gcd(7178655232, 1426532525) => 997'''
    if a < b: a, b = b, a
    while b != 0:
        a, b = b, a % b
    return a

def gcd_extended(a:int, b:int) -> tuple[int, int, int]:
    '''Implementa recursivamente o cálculo do MDC entre a e b
    usando o algoritmo de Euclides. Retorna x, y e d, tais que
    a*x + b*y = d, onde d é o MDC entre a e b. 
    Complexidade: O(log(min(a, b))).
    Exemplo: gcd_extended(7178655232, 1426532525) => (997, -39329, 197913)
    '''
    if a == 0: return b, 0, 1
    d, x, y = gcd_extended(b % a, a)
    return d, y - b // a * x , x

def prod(nums: list[int]) -> int:
    '''Retorna o produto dos elementos em `nums`.
    Exemplo: prod([4, 2, 7]) => 56'''
    total = 1
    for n in nums:
        total *= n
    return total

def poly(*coefficients: list[int]):
    '''
    Cria uma função polinomial baseada nos coeficientes passados e a retorna.
    Exemplo:
    f = poly(1, 2, 3)
    Cria uma função que avalia o polinômio x² + 2x + 3 em qualquer valor de x.
    f(2) => 11
    f(3) => 18
    f(-1) => 2
    '''
    def p(x: int):
        acc = coefficients[0]
        for c in coefficients[1:]:
            acc *= x
            acc += c
        return acc
    return p

def oddify(n:int) -> tuple[int, int]:
    '''Retorna k, q tais que n = 2^k * q, com q ímpar.
    Complexidade:O(log(n)).
    Exemplo: oddify(41) => (3, 5)
    '''
    q = n
    k = 0
    while q % 2 == 0:
        q >>= 1
        k += 1
    return k, q
