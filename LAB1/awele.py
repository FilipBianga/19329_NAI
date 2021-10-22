"""
Autorzy: Filip Bianga
Zasady: https://www.kurnik.pl/oware/zasady.phtml
Gra zwana również jako Oware, Awale, Mankala
"""
"""
Zasady instalacji środowiska:

Jeżeli posiadasz zainstalowanego "pip" w terminalu wpisz poniższą komende

sudo pip install easyAI

W przeciwnym razie pobierz kod źródłowy - https://github.com/Zulko/easyAI -
rozpakuj wszystko do jednego foldewru i w terminalu wpisz

sudo python setup.py install
"""

try:
    import numpy as np
except ImportError:
    print("Sorry, this example requires Numpy installed !")
    raise

from easyAI import TwoPlayerGame


class Awele(TwoPlayerGame):

    def __init__(self, players):
        for i, player in enumerate(players):
            player.score = 0
            player.isstarved = False
            player.camp = i
        self.players = players

        # Building a board
        self.board = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

        self.current_player = 1  # player 1 starts.

    def make_move(self, move):
        if move == "None":
            self.player.isstarved = True
            s = 6 * self.opponent.camp
            self.player.score += sum(self.board[s : s + 6])
            return

        move = "abcdefghijkl".index(move)
# START - why AI always takes points from me from the letters "a", and I from AI always takes points from the letters "g"? Check it out
        pos = move
        for i in range(self.board[move]):
            pos = (pos + 1) % 12
            if pos == move:
                pos = (pos + 1) % 12
            self.board[pos] += 1

        self.board[move] = 0

        while (pos / 6) == self.opponent.camp and (self.board[pos] in [2, 3]):
            self.player.score += self.board[pos]
            self.board[pos] = 0
            pos = (pos - 1) % 12
# END
    def possible_moves(self):
        """
        A player must play any hole that contains enough seeds to
        'feed' the opponent.
        """

        if self.current_player == 1:
            if max(self.board[:6]) == 0:
                return ["None"]
            moves = [i for i in range(6) if (self.board[i] >= 6 - i)]
            if moves == []:
                moves = [i for i in range(6) if self.board[i] != 0]
        else:
            if max(self.board[6:]) == 0:
                return ["None"]
            moves = [i for i in range(6, 12) if (self.board[i] >= 12 - i)]
            if moves == []:
                moves = [i for i in range(6, 12) if self.board[i] != 0]

        return ["abcdefghijkl"[u] for u in moves]

    def show(self):
        """ Print board and show score """
        print("Score: %d / %d" % tuple(p.score for p in self.players))
        print("  ".join("lkjihg"))
        print(" ".join(["%02d" % i for i in self.board[-1:-7:-1]]))
        print(" ".join(["%02d" % i for i in self.board[:6]]))
        print("  ".join("abcdef"))

    def lose(self):
        return self.opponent.score > 24

    def is_over(self):
        return self.lose() or sum(self.board) < 7 or self.opponent.isstarved


if __name__ == "__main__":
    # In what follows we setup the AI and launch a HP-vs-AI match.

    from easyAI import Human_Player, AI_Player, Negamax

    scoring = lambda game: game.player.score - game.opponent.score
    ai = Negamax(6, scoring)
    game = Awele([Human_Player(), AI_Player(ai)])

    game.play()

    if game.player.score > game.opponent.score:
        print("Player %d wins." % game.current_player)
    elif game.player.score < game.opponent.score:
        print("Player %d wins." % game.opponent_index)
    else:
        print("Looks like we have a draw.")