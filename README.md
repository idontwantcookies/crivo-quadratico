# Crivo Quadrático e Teoria dos Números

> Implementação didática, do zero e sem dependências pesadas, dos algoritmos clássicos
> de teoria dos números computacional — culminando no **Crivo Quadrático**, o método
> subexponencial de fatoração de inteiros grandes.

<p>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white">
  <img alt="Licença GPL-3.0" src="https://img.shields.io/badge/Licen%C3%A7a-GPL--3.0-green">
  <img alt="Testes: pytest" src="https://img.shields.io/badge/Testes-pytest-0A9EDC?logo=pytest&logoColor=white">
</p>

**[🔗 Demonstração interativa no navegador](https://idontwantcookies.github.io/crivo-quadratico/)** —
roda o código deste repositório direto no seu navegador, via WebAssembly, sem instalar nada.

---

## Sobre o projeto

Trabalho prático da disciplina de Álgebra A. A proposta é implementar à mão, sem recorrer
a bibliotecas de álgebra computacional, a cadeia completa de algoritmos que leva à
fatoração de inteiros grandes:

```
aritmética modular  →  primalidade  →  fatoração  →  álgebra linear sobre GF(2)  →  crivo quadrático
```

Tudo é Python puro sobre o `int` nativo. A única dependência externa é o `sympy`, usado
apenas como **oráculo independente nos testes** (e numa função auxiliar de núcleo sobre
os racionais que o crivo não utiliza).

## Início rápido

```bash
git clone https://github.com/idontwantcookies/ubiquitous-octo-funicular.git
cd ubiquitous-octo-funicular
pip install -r requirements.txt

python tp2.py
```

O programa pede um inteiro `N` e devolve um fator não trivial dele:

```
Insira o valor de N: 87463

Executando crivo quadrático com primos menores ou iguais a B=43
Fator encontrado para n:  587
Tempo de execução: 1.365ms.
```

A saída traz o limite de suavidade `B` escolhido, o fator encontrado e o tempo total.

### Entrada válida

| Entrada | Comportamento |
| --- | --- |
| `N` composto, `N ≥ 3` | devolve um fator não trivial |
| `N` primo | mensagem de erro e código de saída 1 — não existe fator não trivial |
| `N ∈ {1, 2}` | ⚠️ `ValueError: math domain error` (`find_B` calcula `log(log(N))`) |

## Desempenho

Medições num semiprimo `p × q` com fatores de tamanho equilibrado (o pior caso para
fatoração), em Python puro num único núcleo:

| Dígitos de `N` | Tempo |
| ---: | ---: |
| 14 | 0,01 s |
| 18 | 0,14 s |
| 22 | 0,70 s |
| 24 | 1,7 s |
| 26 | 4,2 s |
| 28 | 15 s |
| 30 | 25 s |

Trinta dígitos é o limite prático confortável desta implementação. O `timeout` padrão de
15 s (parâmetro de `quadratic_sieve`) é verificado entre blocos do crivo, então não é uma
barreira rígida: entradas maiores podem ultrapassá-lo antes de terminar.

A complexidade do crivo quadrático é subexponencial, `O(e^√(ln N · ln ln N))` — assintoticamente
muito melhor que a divisão por tentativa `O(√N)`, ainda que pior que polinomial.

## Como funciona o crivo

O objetivo é achar `a` e `b` com `a² ≡ b² (mod N)` e `a ≢ ±b`. Daí `gcd(a − b, N)` é um
fator não trivial de `N` com probabilidade ~1/2.

1. **Base de fatores** — escolhe-se `B = L(N)^e` e tomam-se os primos `p ≤ B`. O critério
   de Euler descarta os `p` para os quais `N` não é resíduo quadrático: eles nunca
   dividiriam `x² − N`.
2. **Raízes modulares** — resolve-se `x² ≡ N (mod p)` por Tonelli-Shanks. É o que separa o
   crivo do método de Dixon: sabemos de antemão que `p` divide `Q(x) = x² − N` exatamente
   nas posições `x ≡ r (mod p)`, então não se testa candidato por candidato.
3. **Peneiramento** — varre-se um intervalo dividindo de fato cada `Q(x)` pelos primos nas
   posições previstas. Sobra `1` ⟹ `Q(x)` é `B`-suave, e temos uma relação. Sem limiares
   logarítmicos: nenhum falso positivo ou negativo.
4. **Álgebra linear sobre GF(2)** — monta-se a matriz de paridade dos expoentes. Qualquer
   vetor do núcleo indica um subconjunto de relações cujo produto é um quadrado perfeito.
   A eliminação usa máscaras de bits em inteiros grandes, então o XOR de linhas roda em C.
5. **Extração do fator** — cada vetor do núcleo dá um par `(a, b)`; testa-se `gcd(a − b, N)`
   até sair um fator não trivial.

> 📚 Referências: [Collier (UFRJ)](https://www.dcc.ufrj.br/~collier/CursosGrad/topicos/CrivoQuadratico.html)
> e [risencrypto](https://risencrypto.github.io/QuadraticSieve/).

## Módulos

| Módulo | Conteúdo |
| --- | --- |
| `base` | Raiz quadrada inteira, `log₁₀` inteiro, MDC e MDC estendido, avaliação de polinômios. |
| `modular_arithmetic` | Inverso modular, exponenciação binária, ordem, subgrupos, Teorema Chinês do Resto, símbolo de Legendre, raiz quadrada modular (Tonelli-Shanks) e busca de geradores. |
| `primality` | Teste de Miller-Rabin e crivo de Eratóstenes. |
| `factorization` | Função totiente, Pollard rho e decomposição em potências de primos. |
| `linalg` | Operações com vetores e matrizes, RREF sobre os racionais e sobre GF(2), e cálculo de núcleo. A versão GF(2) usa máscaras de bits. |
| `quadratic_sieve` | O crivo propriamente dito: escolha de `B`, raízes modulares, peneiramento por blocos e montagem do sistema. |
| `discrete_log` | Baby-step giant-step e Pohlig-Hellman para o problema do logaritmo discreto. |
| `rsa` | Geração de chaves, cifra e decifra — aplicação direta dos módulos acima. |
| `util` | Cronômetro e exceções do domínio. |

Há ainda `src/generator.cpp`, um utilitário em C++ com GMP que gerou os casos de teste de
referência em `big_numbers/` (compile com `make`; exige `libgmp`).

## Testes

```bash
pytest                      # suíte completa
pytest -m "not slow"        # pula os semiprimos grandes
make test                   # roda com cobertura e gera htmlcov/
```

A suíte cobre cada módulo e usa `sympy` como oráculo independente para validar a RREF e
as fatorações. Alguns algoritmos são probabilísticos (Miller-Rabin, Pollard rho, busca de
geradores), então os testes que dependem de sorteio repetem cada caso várias vezes em vez
de confiar numa única execução.

Testes marcados como `slow` fatoram semiprimos grandes e levam dezenas de segundos.

## Demonstração no navegador

A pasta `docs/` contém uma página estática que executa **o código-fonte real** de `src/`
dentro do navegador, via [PyScript](https://pyscript.net) e Pyodide (CPython compilado
para WebAssembly). Nada foi reescrito em JavaScript e não há back-end: os módulos são
buscados direto do repositório em tempo de execução.

A página demonstra o teste de primalidade de Miller-Rabin, com o veredito rodada a rodada,
e o logaritmo discreto por Pohlig-Hellman, com a fatoração de `p − 1` e a verificação do
resultado.

Para publicar num fork: **Settings → Pages → Deploy from a branch**, apontando para a
pasta `/docs`. Se mudar o nome da branch, atualize o caminho `{SRC}` em `docs/pyscript.json`.

## Estrutura

```
├── src/                  # módulos da biblioteca
├── tests/                # suíte pytest
├── big_numbers/          # casos de teste de referência (gerados via GMP)
├── docs/                 # demonstração web (PyScript + Pyodide)
├── tp2.py                # CLI: lê N e fatora
├── makefile              # atalhos de teste, lint e build do gerador C++
└── requirements.txt
```

## Autores

- Felipe Ribas Muniz
- Chrystian Paulo Ferreira de Melo
- João Marcos Rezende
- Sanny Cristiane Moreira de Sales

## Licença

Distribuído sob a [GNU General Public License v3.0](LICENSE).
