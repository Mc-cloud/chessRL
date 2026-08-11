import chess

class Game:
    def __init__(self, agent_white, agent_black):
        self.agent_white = agent_white
        self.agent_black = agent_black

    def play_game(self, display = True):

        board = chess.Board()

        while not board.is_game_over():
            if display:
                print("")
                print("\n" + "="*30)
                print(board)
                print("="*30)
                tour_str = "Blancs" if board.turn == chess.WHITE else "Noirs"
                print(f"Trait aux {tour_str}...")

            if board.turn == chess.WHITE:
                move = self.agent_white.get_move(board)
                agent_name = self.agent_white.name
            else:
                move = self.agent_black.get_move(board)
                agent_name = self.agent_black.name

            if display:
                print(f"🤖 {agent_name} joue : {board.san(move)}")

            board.push(move)

        if display:
            print("\n🏁 PARTIE TERMINÉE 🏁")
            print(board)
            print(f"Résultat : {board.result()} ({self._get_outcome(board)})")

        return board.result()

    def _get_outcome(self, board: chess.Board):

        if board.is_checkmate():
            return "Mat"
        elif board.is_stalemate():
            return "Pat"
        elif board.is_insufficient_material():
            return "Matériel insuffisant (Nul)"
        else:
            return "Fin de partie"