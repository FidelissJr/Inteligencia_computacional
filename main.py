from src.ex1 import ex1
from src.ex2 import ex2
from src.ex3 import ex3
from src.ex4 import ex4
from src.ex7 import ex7


def main() -> None:
    # Descomente o exercício que deseja executar
    # ex1()
    # ex2()
    # ex3()

    # EX4 - N-Rainhas: execute apenas UM tabuleiro por vez (descomente o config desejado)
    # ex4("./config/config_ex4_8.json")
    # ex4("./config/config_ex4_16.json")
    # ex4("./config/config_ex4_32.json")
    # ex4("./config/config_ex4_64.json")
    # ex4("./config/config_ex4_128.json")

    # EX7 - N-Rainhas em tabuleiro valorado (execute um config por vez)
    # ex7("./config/config_ex7_8.json")
    ex7("./config/config_ex7_16.json")


if __name__ == "__main__":
    main()
