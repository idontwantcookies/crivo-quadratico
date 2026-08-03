import sys

from src.quadratic_sieve import quadratic_sieve, find_B
from src.util import SieveTimeout, Timer

n = (int(input("Insira o valor de N: ")))

print()

with Timer():
    B = find_B(n)
    print(f"Executando crivo quadrático com primos menores ou iguais a B={B}")
    try:
        d = quadratic_sieve(n)
        print("Fator encontrado para n: ", d)
    except (SieveTimeout, RuntimeError) as e:
        # A biblioteca levanta exceção; quem decide encerrar o processo é o CLI.
        print(e)
        sys.exit(1)
