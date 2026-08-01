import chess

class Base_agent :
    def __init__(self, name = "Unnamed"):
        self.name = name

    def get_move(self, board : chess.Board) :

        raise NotImplementedError("on doit implémanter la méthode pour chaque agent")

