#!/usr/bin/env python

## Exercício 1: Uma rede que descobre se é V ou Λ

from functools import reduce

def sign (x):
    """Função que pega sinal"""
    if x > 0:   return 1
    else:       return -1

def lêTeste (nomeArquivo):
    with open (nomeArquivo, 'r') as arquivo:
        return int (arquivo.readline ()), [int (num) for num in arquivo.read ().split ()]

def lêEntrada (nomeArquivo):
    with open (nomeArquivo, 'r') as arquivo:
        return [int (num) for num in arquivo.read ().split ()]

class Adaline:
    """Rede adaline, que aprende e deduz =P"""
    def __init__ (self, taxaAprendizado = 0.5):
        self.taxaAprendizado = taxaAprendizado

    def treina (self, nomesArquivos):
        """Treina a rede, a partir de vários arquivos contendo resultado
        esperado e valores. Cada treino zera os pesos, então cuidado!"""
        respostaEsperada = []
        entrada = []
        for nome in nomesArquivos:
            resp, entr = lêTeste (nome)
            respostaEsperada.append (resp)
            entrada.append (entr)

        self.pesos = [0] * len (entrada[0])

        # loop dos ciclos de treinamento
        cabou = False
        iterações = 0
        while not cabou:
            cabou = True
            iterações += 1
            for s_out, entradaAtual in zip (respostaEsperada, entrada):
                # resultado é a soma dos peso * entrada
                # resposta deu diferente: ajusta pesos e vai ter que fazer denovo
                resultado = self.reconhece (entradaAtual)
                if resultado != s_out:
                    cabou = False
                    for i in range (len (self.pesos)):
                        self.pesos[i] += self.taxaAprendizado * (s_out - resultado) * entradaAtual[i]
        print ('Em', iterações, 'iterações:', self.pesos)

    def reconhece (self, entrada):
        """Reconhece uma entrada, retornando a função lá do sinal da soma dos
        pesos vezes as entradas"""
        resultado = 0
        for i, x in enumerate (entrada):
            resultado += self.pesos[i] * x
        return sign (resultado)



def main ():
    A = Adaline ()
    A.treina (['treino1.txt', 'treino2.txt'])
    print (A.reconhece (lêEntrada ('teste2.txt')))

if __name__ == '__main__':
    main ()
