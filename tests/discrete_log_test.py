import pytest

from src import discrete_log
from src.factorization import pollard_rho_prime_power_decomposition
from src.modular_arithmetic import is_generator, powmod
from tests.big_numbers import bsgs


@pytest.mark.parametrize("g,x,h,p", bsgs)
def test_baby_step_giant_step(g, x, h, p):
    x = discrete_log.baby_step_giant_step(g, h, p, p - 1)
    assert x != 0 and x is not None
    assert g**x % p == h

def test_pohlig_hellman():
    n = 101
    f = {2: 2, 5: 2}
    g, h = 15, 100
    assert discrete_log.pohlig_hellman(g, h, n, f) == 50


def test_pohlig_hellman_prime_power():
    g, h, p, e, o = 27, 40, 2, 3, 41
    assert discrete_log.pohlig_hellman_prime_power_order(g, h, p, e, o) == 4


# --------------------------------------------------------------------------
# Regressão de integração: com a fatoração errada de p-1, o Pohlig-Hellman
# recebia uma ordem de grupo menor que a real e falhava lá dentro, com
# "Baby-step, giant-step failed: g does not generate n" — apontando para um g
# que É um gerador legítimo. A mensagem culpava o módulo errado; a causa
# estava em pollard_rho_prime_power_decomposition.
#
# Estes casos exigem que p-1 tenha alguma potência de primo >= 4, que é a
# condição para o bug de multiplicidade aparecer.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("p,g,x", [
    [8763841, 4446332, 4948153],    # p-1 = 2**6 * 3**2 * 5 * 17 * 179
    [8763841, 4446332, 1],
    [8763841, 4446332, 8763839],    # x = phi - 1, extremo do intervalo
    [40961, 3, 12345],              # p-1 = 2**13 * 5
    [65537, 3, 4242],               # p-1 = 2**16, potência de primo pura
])
def test_pohlig_hellman_com_potencia_de_primo_alta(p, g, x):
    f = pollard_rho_prime_power_decomposition(p - 1)

    produto = 1
    for pp, ee in f.items():
        produto *= pp**ee
    assert produto == p - 1, f"fatoração de p-1 saiu errada: {dict(f)}"

    assert is_generator(g, p, p - 1, f), "caso de teste mal construído: g não é gerador"

    h = powmod(g, x, p)
    assert discrete_log.pohlig_hellman(g, h, p, f) == x
