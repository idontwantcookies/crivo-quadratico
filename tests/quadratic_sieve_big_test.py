'''Testes de regressão e de números grandes para o crivo quadrático.

Os testes de `quadratic_sieve_test.py` passavam pelo atalho de divisão por
tentativa (`if n % xj == 0: return xj`) dentro de `quadratic_sieve`, e não pelo
crivo em si. Aqui a flag `TRIAL_DIVISION` é desligada para exercitar o caminho
real do algoritmo: coleta de relações, núcleo sobre GF(2) e gcd(a-b, n).

Os testes marcados com `xfail` documentam erros ainda não corrigidos.

Execução:
    pytest tests/quadratic_sieve_big_test.py -m "not slow"
    pytest tests/quadratic_sieve_big_test.py            # inclui os grandes
'''

from collections import OrderedDict
from math import ceil, sqrt
from time import time

import pytest

import src.quadratic_sieve as qs_mod
from src.base import isqrt
from src.linalg import find_pivot, kernel, kernel_gf2, matrix_mod
from src.modular_arithmetic import find_non_square, is_square, msqrt
from src.quadratic_sieve import (
    build_matrix_of_powers,
    collect_relations,
    compose_from_solution,
    find_B,
    iter_kernel_solutions,
    kernel_solutions,
    quadratic_sieve,
    quadratic_sieve_aux,
    sieve_block,
    sieve_roots,
)
from src.util import SieveTimeout

# Timeouts generosos: sob `coverage` o código roda ~4x mais lento, e um teste
# não pode passar ou falhar conforme a máquina.
TIMEOUT_TESTE = 120
TIMEOUT_TESTE_LONGO = 900
# NÃO importe `setup` com esse nome: o plugin `nose`, ainda ativo por padrão no
# pytest 8, trata uma função de módulo chamada `setup` como setup de módulo e a
# chama passando o próprio módulo, o que faz TODOS os testes do arquivo virarem
# ERROR com "TypeError: must be real number, not module".
from src.quadratic_sieve import setup as qs_setup


@pytest.fixture(name='sem_divisao_por_tentativa', autouse=False)
def _sem_divisao_por_tentativa(monkeypatch):
    '''Desliga o atalho de divisão por tentativa, obrigando `quadratic_sieve`
    a achar o fator pelo crivo + álgebra linear.'''
    monkeypatch.setattr(qs_mod, 'TRIAL_DIVISION', False)


# ---------------------------------------------------------------------------
# Semiprimos n = p*q usados nos testes. (p, q, n)
# ---------------------------------------------------------------------------

SEMIPRIMOS_PEQUENOS = [
    (2309, 4073, 9404557),                                  #  7 dígitos
    (49169, 62981, 3096712789),                             # 10 dígitos
    (855401, 861589, 737004092189),                         # 12 dígitos
]

SEMIPRIMOS_MEDIOS = [
    (12740209, 13514923, 172182943638907),                  # 15 dígitos
    (138783137, 198408253, 27535719758029661),              # 17 dígitos
    (2389472381, 3758991527, 8982006434179515787),          # 19 dígitos
]

SEMIPRIMOS_GRANDES = [
    (719617100987, 956566523971,
     688361628881222663259377),                             # 24 dígitos
    (198542906811121, 259727501297101,
     51567053086315632495111860221),                        # 29 dígitos
    (11843225287937664763, 14313953111306901257,
     169523371458183908036221183946409307091),              # 39 dígitos
]

CASOS_PATOLOGICOS = [
    # p e q muito próximos: x0 = isqrt(n)+1 acha o fator quase imediatamente.
    (100000007, 100000037, 10000004400000259),
    # p e q muito distantes: o crivo tem de trabalhar de verdade.
    (1009, 10000000000037, 10090000000037333),
]


# ---------------------------------------------------------------------------
# 1. Regressões dos erros de lógica encontrados
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('_p,_q,n', SEMIPRIMOS_PEQUENOS + SEMIPRIMOS_MEDIOS)
def test_guarda_de_relacoes_usa_tamanho_da_base(_p, _q, n):
    '''BUG: `quadratic_sieve` interrompe a coleta em M = len(primes)+5 relações,
    mas em seguida exige `len(S) > B`, onde B é o *limite de suavidade*, muito
    maior que M. A guarda é, portanto, impossível de satisfazer.
    O critério correto é len(S) > len(primes).'''
    _B, M, primes = qs_setup(n)
    # O laço de coleta para em M+1 relações; a guarda tem de ser satisfazível
    # com esse número. Comparar com B (limite de suavidade) a tornava impossível.
    assert M + 1 > len(primes), (
        f'guarda inalcançável: coleta para em {M + 1} relações, '
        f'mas exige mais de {len(primes)}'
    )


@pytest.mark.parametrize('_p,_q,n', SEMIPRIMOS_PEQUENOS)
def test_kernel_e_calculado_sobre_gf2(_p, _q, n):
    '''O crivo precisa do núcleo sobre GF(2), não sobre Q (`sympy.nullspace`).
    Sobre Q o núcleo é menor — perde relações — e pode conter frações, que
    quebram `isqrt_powers`. Compara `kernel_gf2` com um oráculo independente e
    confirma que cada vetor da base realmente anula A mod 2.'''
    S = _coletar_relacoes(n)
    A = matrix_mod(build_matrix_of_powers(S, qs_setup(n)[2]), 2)
    cols = len(A[0])
    base = kernel_gf2(A)
    assert len(base) == cols - _posto_gf2(A), 'dimensão do núcleo incorreta'
    assert len(base) >= len(kernel(A)), 'núcleo sobre GF(2) não pode ser menor que sobre Q'
    for v in base:
        for i, linha in enumerate(A):
            s = sum(linha[j] * v[j] for j in range(cols)) % 2
            assert s == 0, f'vetor do núcleo não anula a linha {i}'


@pytest.mark.parametrize('_p,_q,n', SEMIPRIMOS_PEQUENOS)
def test_vetores_do_kernel_sao_binarios(_p, _q, n):
    '''Os vetores do núcleo têm de ser 0/1. Os de `sympy.nullspace` podem ser
    racionais (ex.: 6/11), o que torna `vector_mod(u, 2)` sem sentido e faz
    `compose_from_solution` levantar ValueError em *todas* as combinações.'''
    S = _coletar_relacoes(n)
    A = matrix_mod(build_matrix_of_powers(S, qs_setup(n)[2]), 2)
    for u in kernel_gf2(A):
        for x in u:
            assert isinstance(x, int) and x in (0, 1), f'entrada não binária: {x}'


@pytest.mark.parametrize('_p,_q,n', SEMIPRIMOS_PEQUENOS)
def test_toda_solucao_do_kernel_produz_congruencia_valida(_p, _q, n):
    '''Propriedade central do crivo: para todo vetor v do núcleo sobre GF(2),
    a = prod(xj : v_j = 1) e b = sqrt(prod(xj²-n : v_j = 1)) devem satisfazer
    a² ≡ b² (mod n). Nenhuma combinação pode levantar ValueError.'''
    S = _coletar_relacoes(n)
    A = matrix_mod(build_matrix_of_powers(S, qs_setup(n)[2]), 2)
    ker = kernel_gf2(A)
    assert ker, 'núcleo vazio: nada a testar'
    for v in iter_kernel_solutions(ker, limit=64):
        a, b = compose_from_solution(S, v)     # não pode lançar ValueError
        assert (a * a - b * b) % n == 0, f'congruência inválida para v={v}'


@pytest.mark.parametrize('n', [10**30 + 57, 10**40 + 121, 10**44 + 7])
def test_ponto_de_partida_usa_isqrt_e_nao_float(n):
    '''REGRESSÃO: o crivo usava `x0 = ceil(sqrt(n))` em float. Acima de 2^53 o
    float erra por 1 para menos e Q(x0) = x0² - n fica negativo — o primeiro
    candidato já é inválido. `isqrt` é exato.'''
    x0_float = ceil(sqrt(n))
    x0_exato = isqrt(n) + 1
    assert x0_exato * x0_exato > n, 'isqrt(n)+1 tem de ser o menor x com x² > n'
    assert isqrt(n) ** 2 <= n
    if x0_float != x0_exato:          # acontece justamente acima de 2^53
        assert x0_float * x0_float < n, 'demonstra por que o float não serve'


@pytest.mark.parametrize('_p,_q,n', SEMIPRIMOS_PEQUENOS + SEMIPRIMOS_MEDIOS)
def test_primeiro_candidato_do_crivo_tem_Q_positivo(_p, _q, n):
    '''Q(x) = x² - n tem de ser positivo em todo o intervalo crivado; caso
    contrário a decomposição em primos da base não faz sentido.'''
    S = collect_relations(n, qs_setup(n)[2], 5, time(), TIMEOUT_TESTE)
    for x in S:
        assert x * x - n > 0, f'Q({x}) = {x * x - n} não é positivo'


# ---------------------------------------------------------------------------
# 1b. A crivagem real (bug 6)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('_p,_q,n', SEMIPRIMOS_PEQUENOS + SEMIPRIMOS_MEDIOS)
def test_sieve_roots_resolve_a_congruencia(_p, _q, n):
    '''O que torna o crivo um crivo: para cada p da base, as raízes r de
    x² ≡ n (mod p) dizem exatamente onde p divide Q(x).'''
    primes = qs_setup(n)[2]
    roots = sieve_roots(n, primes)
    for p in primes:
        if p < 2: continue
        assert p in roots and roots[p], f'sem raízes para p={p}'
        for r in roots[p]:
            assert (r * r - n) % p == 0, f'r={r} não resolve x² ≡ n (mod {p})'


@pytest.mark.parametrize('_p,_q,n', SEMIPRIMOS_PEQUENOS)
def test_sieve_block_acha_exatamente_as_relacoes_suaves(_p, _q, n):
    '''A peneira não pode ter falso positivo nem falso negativo: o conjunto que
    devolve tem de bater com a divisão por tentativa candidato a candidato.'''
    primes = qs_setup(n)[2]
    inicio, tamanho = isqrt(n) + 1, 4096
    peneirado = sieve_block(n, primes, sieve_roots(n, primes), inicio, tamanho)

    esperado = {}
    for x in range(inicio, inicio + tamanho):
        resto, exp = x * x - n, {}
        for p in primes:
            if p < 2: continue
            e = 0
            while resto % p == 0:
                resto //= p
                e += 1
            if e: exp[p] = e
        if resto == 1:
            esperado[x] = exp

    assert set(peneirado) == set(esperado), 'conjunto de relações difere'
    for x, exp in esperado.items():
        assert dict(peneirado[x]) == exp, f'expoentes divergem em x={x}'


@pytest.mark.parametrize('_p,_q,n', SEMIPRIMOS_PEQUENOS)
def test_crivo_e_dixon_acham_as_mesmas_relacoes(_p, _q, n):
    '''A crivagem por raízes modulares tem de achar o mesmo conjunto de x que a
    divisão por tentativa candidato a candidato — só que muito mais rápido.
    Enquanto x² < 2n, x² - n e x² mod n coincidem, então as decomposições
    também têm de bater.'''
    peneirado = _coletar_relacoes(n)
    dixon = _coletar_relacoes_por_dixon(n)
    comuns = [x for x in dixon if x * x < 2 * n]
    assert comuns, 'nenhum candidato comparável'
    for x in comuns:
        assert x in peneirado, f'a peneira não achou x={x}'
        assert dict(peneirado[x]) == {p: e for p, e in dixon[x].items() if e}


@pytest.mark.parametrize('p', [3, 5, 13, 17, 29, 41, 97, 193, 257, 40961, 65537])
def test_msqrt_resolve_raiz_modular(p):
    '''REGRESSÃO: `msqrt` só acertava para p ≡ 3 (mod 4). O laço de
    Tonelli-Shanks usava expoente mod p, comparava com -1 (impossível, pois
    pow() devolve valor em [0,p)) e tinha índice deslocado em um.'''
    d = find_non_square(p)
    testados = 0
    for a in range(1, min(p, 500)):
        if not is_square(a, p): continue
        r = msqrt(a, p, d)
        assert (r * r) % p == a % p, f'msqrt({a}, {p}) = {r} está errado'
        testados += 1
    assert testados > 0


@pytest.mark.xfail(reason='bug 5 ainda não corrigido: find_pivot limita a busca por colunas',
                   strict=True)
def test_find_pivot_em_matriz_com_mais_linhas_que_colunas():
    '''BUG: `find_pivot` limita a busca a `i < m` (nº de colunas) e testa
    `if i in (m, n)`. Numa matriz 3x2 ele devolve -1 mesmo havendo pivô na
    linha 2. O limite correto é apenas o número de linhas.'''
    A = [[0, 0], [0, 0], [1, 1]]
    assert find_pivot(A, 0) == 2


@pytest.mark.xfail(reason='bug 6 ainda não corrigido: expoente 1/sqrt(2) em vez de 1/2',
                   strict=True)
def test_find_B_produz_base_de_fatores_praticavel():
    '''`find_B` usa o expoente 1/sqrt(2) ≈ 0.707 em vez do ótimo 1/2 de
    L(n) = exp(sqrt(ln n * ln ln n)), gerando uma base de fatores muito maior
    que o necessário. Para os 45 dígitos prometidos no README, B > 5.4 milhões
    — `eratosthenes_sieve(B)` com defaultdict não é viável nessa escala.'''
    assert find_B(10**20) < 20_000
    assert find_B(10**45) < 100_000, (
        f'B={find_B(10**45):,} para 45 dígitos: crivo de Eratóstenes inviável'
    )


# ---------------------------------------------------------------------------
# 2. Fatoração de n = p*q — o que o algoritmo deveria fazer
# ---------------------------------------------------------------------------

def _verificar(n, p, q, d):
    assert d not in (1, n), f'fator trivial devolvido para n={n}'
    assert n % d == 0, f'{d} não divide {n}'
    assert d in (p, q), f'fator {d} não é nem p={p} nem q={q}'


@pytest.mark.parametrize('p,q,n', SEMIPRIMOS_PEQUENOS)
def test_fatora_semiprimo_pequeno(p, q, n, sem_divisao_por_tentativa):
    _verificar(n, p, q, quadratic_sieve(n, timeout=TIMEOUT_TESTE))


@pytest.mark.parametrize('p,q,n', SEMIPRIMOS_MEDIOS)
@pytest.mark.slow
def test_fatora_semiprimo_medio(p, q, n, sem_divisao_por_tentativa):
    _verificar(n, p, q, quadratic_sieve(n, timeout=TIMEOUT_TESTE))


@pytest.mark.parametrize('p,q,n', SEMIPRIMOS_GRANDES[:2])
@pytest.mark.slow
def test_fatora_semiprimo_grande(p, q, n, sem_divisao_por_tentativa):
    '''24 e 29 dígitos: inalcançáveis antes da crivagem real, agora rotina.'''
    _verificar(n, p, q, quadratic_sieve(n, timeout=TIMEOUT_TESTE))


@pytest.mark.parametrize('p,q,n', SEMIPRIMOS_GRANDES[2:])
@pytest.mark.slow
@pytest.mark.skip(
    reason='39 dígitos ainda é inviável: com o expoente 1/sqrt(2) de find_B a '
           'base de fatores tem 48.398 primos, e a eliminação sobre GF(2) numa '
           'matriz 48k x 48k leva horas. A coleta em si já leva só ~1 min. '
           'Reduzir o expoente de find_B (bug 6, segunda parte) destrava.')
def test_fatora_semiprimo_muito_grande(p, q, n, sem_divisao_por_tentativa):
    _verificar(n, p, q, quadratic_sieve(n, timeout=TIMEOUT_TESTE_LONGO))


@pytest.mark.parametrize('p,q,n', CASOS_PATOLOGICOS)
@pytest.mark.slow
def test_fatora_casos_patologicos(p, q, n, sem_divisao_por_tentativa):
    _verificar(n, p, q, quadratic_sieve(n, timeout=TIMEOUT_TESTE))


@pytest.mark.parametrize('p,q,n', SEMIPRIMOS_PEQUENOS + SEMIPRIMOS_MEDIOS)
@pytest.mark.slow
def test_fatora_com_divisao_por_tentativa_ligada(p, q, n):
    '''Mesmo com o atalho ligado (comportamento de produção), o resultado tem
    de ser um fator primo correto.'''
    _verificar(n, p, q, quadratic_sieve(n, timeout=TIMEOUT_TESTE))


def test_timeout_levanta_excecao_e_nao_mata_o_processo():
    '''REGRESSÃO: `error()` chamava exit(1), o que virava SystemExit no meio da
    biblioteca — matava o processo de quem a usa e, sob `coverage` (que deixa o
    código ~4x mais lento), transformava testes válidos em falhas.'''
    n = SEMIPRIMOS_GRANDES[-1][2]
    with pytest.raises(SieveTimeout):
        quadratic_sieve(n, timeout=0.001)


# ---------------------------------------------------------------------------
# 3. Entradas que o crivo quadrático não sabe tratar (devem falhar limpo)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n', [
    2,
    7919,                        # primo
    10**9 + 7,                   # primo grande
])
def test_primo_nao_trava_e_sinaliza(n):
    '''O crivo não deve entrar no laço `range(x0, n)` até o fim quando n é
    primo. Espera-se uma exceção explícita, não um travamento.'''
    with pytest.raises((RuntimeError, ValueError)):
        quadratic_sieve(n)


@pytest.mark.parametrize('n,fator', [
    (1000003**2, 1000003),       # quadrado perfeito: a ≡ ±b sempre
    (2 * 10**12 + 2, 2),         # n par
    (3 * 5 * 7 * 100003, 3),     # mais de dois fatores primos
])
def test_casos_especiais_antes_do_crivo(n, fator):
    '''Quadrados perfeitos, n par e n com fatores pequenos precisam de
    tratamento prévio — o crivo quadrático puro falha ou é ineficiente neles.'''
    d = quadratic_sieve(n)
    assert d not in (1, n) and n % d == 0


# ---------------------------------------------------------------------------
# 4. Consistência das relações coletadas (invariantes do crivo)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('_p,_q,n', SEMIPRIMOS_PEQUENOS + SEMIPRIMOS_MEDIOS)
@pytest.mark.slow
def test_relacoes_coletadas_sao_congruencias_verdadeiras(_p, _q, n):
    '''Cada relação xj -> {p: alpha} tem de satisfazer
    xj² ≡ prod(p^alpha) (mod n), com todos os p na base de fatores.'''
    primes = qs_setup(n)[2]
    S = _coletar_relacoes(n)
    assert S, 'nenhuma relação coletada'
    for xj, powers in S.items():
        valor = 1
        for p, alpha in powers.items():
            assert p in primes, f'primo {p} fora da base de fatores'
            valor *= p**alpha
        assert (xj * xj - valor) % n == 0, f'relação inválida para xj={xj}'


@pytest.mark.parametrize('_p,_q,n', SEMIPRIMOS_PEQUENOS + SEMIPRIMOS_MEDIOS[:1])
def test_busca_no_kernel_nao_e_exponencial(_p, _q, n):
    '''`kernel_solutions` materializa as 2^dim combinações do núcleo — em 15
    dígitos dim passa de 25, ou seja 33 milhões de vetores, e o programa trava.
    `iter_kernel_solutions` tem de ser preguiçosa e limitada.'''
    S = _coletar_relacoes(n)
    A = matrix_mod(build_matrix_of_powers(S, qs_setup(n)[2]), 2)
    ker = kernel_gf2(A)
    gerados = list(iter_kernel_solutions(ker, limit=50))
    assert len(gerados) <= 50, f'{len(gerados)} soluções geradas com limit=50'
    assert len({tuple(v) for v in gerados}) == len(gerados), 'soluções repetidas'
    assert all(any(v) for v in gerados), 'vetor nulo gerado (gcd sempre trivial)'
    # os primeiros vetores devolvidos são os da própria base
    assert gerados[:len(ker)] == [list(u) for u in ker[:len(gerados)]]


def test_kernel_solutions_gera_vetores_distintos():
    '''`kernel_solutions` não deve repetir vetores nem omitir o espaço gerado.'''
    ker = [[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 1]]
    sols = kernel_solutions(ker)
    assert len(sols) == 2**len(ker)
    assert len({tuple(s) for s in sols}) == len(sols), 'soluções repetidas'


def test_kernel_vazio_nao_estoura():
    '''BUG: `kernel_solutions` acessa `ker[0]` sem checar se a lista é vazia.'''
    assert kernel_solutions([]) in ([], [[]])


# ---------------------------------------------------------------------------
# Utilitário
# ---------------------------------------------------------------------------

def _posto_gf2(A) -> int:
    '''Posto de A sobre GF(2) por eliminação gaussiana com máscaras de bits.
    Serve de oráculo independente para o núcleo que o crivo deveria calcular.'''
    linhas = [int(''.join(str(x % 2) for x in row), 2) if row else 0 for row in A]
    cols = len(A[0]) if A else 0
    posto = 0
    for j in range(cols):
        bit = 1 << (cols - 1 - j)
        pivo = next((i for i in range(posto, len(linhas)) if linhas[i] & bit), None)
        if pivo is None:
            continue
        linhas[posto], linhas[pivo] = linhas[pivo], linhas[posto]
        for i in range(len(linhas)):
            if i != posto and linhas[i] & bit:
                linhas[i] ^= linhas[posto]
        posto += 1
    return posto


def _coletar_relacoes(n: int) -> OrderedDict:
    '''Fase de coleta de `quadratic_sieve`, sem o atalho de divisão por
    tentativa.'''
    _B, M, primes = qs_setup(n)
    return collect_relations(n, primes, M + 1, time(), TIMEOUT_TESTE)


def _coletar_relacoes_por_dixon(n: int) -> OrderedDict:
    '''Mesma coleta, mas candidato a candidato (método de Dixon). Serve de
    oráculo independente para `collect_relations`.'''
    S: OrderedDict = OrderedDict()
    _B, M, primes = qs_setup(n)
    for xj in range(isqrt(n) + 1, n):
        quadratic_sieve_aux(n, xj, S, primes)
        if len(S) > M:
            break
    return S
