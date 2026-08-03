"""
Este arquivo é só "cola" de interface: lê os campos da página, chama as
funções REAIS do repositório (buscadas ao vivo de src/*.py — ver
pyscript.json) e escreve o resultado de volta no HTML. Nenhuma lógica de
teoria dos números foi reescrita aqui; ela mora inteiramente em src/.
"""

import asyncio
import platform
import random
import time

from pyscript import web, when

from src.base import ilog10, oddify
from src.discrete_log import pohlig_hellman
from src.factorization import pollard_rho_prime_power_decomposition
from src.modular_arithmetic import find_generator, is_generator, powmod
from src.primality import miller_test, prime_miller_rabin


# ---------------------------------------------------------------- boot ----

web.page["py-version-badge"].innerHTML = f"Python {platform.python_version()} · no navegador"
web.page.body.classes.remove("booting")


def parse_int(raw: str) -> int:
    return int(raw.strip().replace(" ", "").replace(".", "").replace(",", ""))


# ------------------------------------------------------- primalidade ------

MR_MAX_DIGITS = 200


@when("change", "#mr-preset")
def mr_preset_change(event):
    value = event.target.value
    if value:
        web.page["mr-n"].value = value


@when("click", "#mr-run")
async def mr_run(event=None):
    out = web.page["mr-output"]
    button = web.page["mr-run"]

    try:
        n = parse_int(web.page["mr-n"].value)
    except ValueError:
        out.innerHTML = "<p class='error'>Digite um inteiro válido.</p>"
        return

    if n < 2:
        out.innerHTML = "<p class='error'>N precisa ser maior ou igual a 2.</p>"
        return
    if len(str(n)) > MR_MAX_DIGITS:
        out.innerHTML = f"<p class='error'>Use um N com até {MR_MAX_DIGITS} dígitos, pra manter a demo instantânea.</p>"
        return

    button.disabled = True
    out.innerHTML = "<p class='pending'>Calculando…</p>"
    await asyncio.sleep(0.02)  # deixa o navegador repintar antes da chamada síncrona

    try:
        t0 = time.perf_counter()
        is_probable_prime = prime_miller_rabin(n)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        rows_html = ""
        rounds_shown = 0
        if n == 2:
            rows_html = "<tr><td>—</td><td>—</td><td>caso base: 2 é primo por definição</td></tr>"
        else:
            k, q = oddify(n - 1)
            rounds = min(10, max(3, ilog10(n) + 1))
            for i in range(rounds):
                b = random.randint(2, n - 1) if n > 3 else 2
                verdict = miller_test(n, b, k, q)
                rounds_shown += 1
                rows_html += (
                    f"<tr><td>{i + 1}</td><td>{b}</td>"
                    f"<td>{'inconclusivo (talvez primo)' if verdict else 'reprovou → composto'}</td></tr>"
                )
                if not verdict:
                    break

        status = "provavelmente primo" if is_probable_prime else "composto"
        badge = "ok" if is_probable_prime else "bad"

        out.innerHTML = f"""
          <div class="result {badge}">{n} é <strong>{status}</strong></div>
          <p class="meta">prime_miller_rabin(n) · {elapsed_ms:.3f} ms</p>
          <table class="trace">
            <thead><tr><th>#</th><th>base b</th><th>miller_test(n, b, k, q)</th></tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
          <p class="meta">
            {rounds_shown or 1} rodada(s) de <code>miller_test</code> mostradas acima
            (nova amostra de bases, só para ilustrar — o veredito final vem de
            <code>prime_miller_rabin</code>).
          </p>
        """
    except Exception as exc:  # noqa: BLE001 - mostra qualquer erro real da lib ao usuário
        out.innerHTML = f"<p class='error'>{type(exc).__name__}: {exc}</p>"
    finally:
        button.disabled = False


# ------------------------------------------------------ logaritmo log -----

DLOG_MAX_P = 2**40


def random_prime(bits: int) -> int:
    while True:
        candidate = random.getrandbits(bits) | 1 | (1 << (bits - 1))
        if prime_miller_rabin(candidate):
            return candidate


@when("click", "#dlog-random")
async def dlog_random(event=None):
    out = web.page["dlog-output"]
    button = web.page["dlog-random"]
    button.disabled = True
    out.innerHTML = "<p class='pending'>Gerando primo, fatorando p−1 e achando um gerador…</p>"
    await asyncio.sleep(0.02)

    try:
        bits = random.choice([16, 20, 24, 28])
        p = random_prime(bits)
        phi = p - 1
        f = pollard_rho_prime_power_decomposition(phi)
        g = find_generator(p, phi, f, timeout=6)
        x = random.randint(1, phi - 1)
        h = powmod(g, x, p)

        web.page["dlog-p"].value = str(p)
        web.page["dlog-g"].value = str(g)
        web.page["dlog-h"].value = str(h)

        fact_str = " · ".join(f"{pp}^{ee}" for pp, ee in sorted(f.items()))
        out.innerHTML = f"""
          <p class="meta">
            Exemplo gerado: p = {p} ({bits} bits), p−1 = {fact_str}.
            O segredo x = {x} foi escondido em h — clique em Resolver pra redescobri-lo.
          </p>
        """
    except Exception as exc:  # noqa: BLE001
        out.innerHTML = f"<p class='error'>{type(exc).__name__}: {exc}</p>"
    finally:
        button.disabled = False


@when("click", "#dlog-run")
async def dlog_run(event=None):
    out = web.page["dlog-output"]
    button = web.page["dlog-run"]

    try:
        p = parse_int(web.page["dlog-p"].value)
        g = parse_int(web.page["dlog-g"].value)
        h = parse_int(web.page["dlog-h"].value)
    except ValueError:
        out.innerHTML = "<p class='error'>p, g e h precisam ser inteiros.</p>"
        return

    if p < 5 or p > DLOG_MAX_P:
        out.innerHTML = f"<p class='error'>Use um p primo entre 5 e 2^40, pra manter a demo instantânea no navegador.</p>"
        return

    button.disabled = True
    out.innerHTML = "<p class='pending'>Verificando se p é primo…</p>"
    await asyncio.sleep(0.02)

    try:
        if not prime_miller_rabin(p):
            out.innerHTML = f"<p class='error'>{p} não é primo — esta implementação de Pohlig-Hellman assume o grupo (ℤ/pℤ)*.</p>"
            return

        g, h = g % p, h % p
        if g in (0, 1):
            out.innerHTML = "<p class='error'>g precisa ser diferente de 0 e 1 (mod p).</p>"
            return

        out.innerHTML = "<p class='pending'>Fatorando p−1…</p>"
        await asyncio.sleep(0.02)
        t0 = time.perf_counter()
        f = pollard_rho_prime_power_decomposition(p - 1)
        fact_str = " · ".join(f"{pp}^{ee}" for pp, ee in sorted(f.items()))

        if not is_generator(g, p, p - 1, f):
            out.innerHTML = f"""
              <p class="error">
                g = {g} não é uma raiz primitiva de p (não gera o grupo inteiro).
                Esta implementação de Pohlig-Hellman exige isso. Tente outro g,
                ou use "🎲 exemplo aleatório".
              </p>
              <p class="meta">p−1 = {fact_str}</p>
            """
            return

        out.innerHTML = f"<p class='pending'>p−1 = {fact_str}. Resolvendo com Pohlig-Hellman…</p>"
        await asyncio.sleep(0.02)
        x = pohlig_hellman(g, h, p, f)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        check = powmod(g, x, p)
        ok = check == h
        badge = "ok" if ok else "bad"

        out.innerHTML = f"""
          <div class="result {badge}">x = {x}</div>
          <p class="meta">p−1 = {fact_str} · {elapsed_ms:.2f} ms</p>
          <p class="verify">verificação: g^x mod p = {check} {'✅ bate com h' if ok else '❌ não bate com h'}</p>
        """
    except Exception as exc:  # noqa: BLE001
        out.innerHTML = f"<p class='error'>{type(exc).__name__}: {exc}</p>"
    finally:
        button.disabled = False
