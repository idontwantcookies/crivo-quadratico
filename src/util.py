# pylint: skip-file

from time import time

Powers = dict[int, int]

class Timer:
    def __enter__(self):
        self.start = time()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.stop = time()
        t = self.stop - self.start
        t = t * 1000
        print(f"Tempo de execução: {t:.3f}ms.")
        print()

class SieveTimeout(Exception):
    '''Levantada quando um algoritmo excede o tempo limite dado.

    Existe para que a biblioteca não chame exit() no meio de um cálculo: isso
    mataria o processo de quem a usa e, nos testes, transforma um estouro de
    tempo em SystemExit no lugar de uma falha tratável.'''


def error(msg:str):
    print(msg)
    exit(1)
