import chess
import random
from .base_agent import Base_agent

class HumanAgent(Base_agent):
    def __init__(self, name="Joueur_Humain"):
        super().__init__(name)

    def get_move(self, board: chess.Board) -> chess.Move:
        # Affiche la liste des coups légaux pour t'aider (optionnel)
        # print("Coups légaux :", [board.uci(m) for m in board.legal_moves])
        
        while True:
            move_str = input(f"{self.name}, entre ton coup (ex: e2e4) : ").strip()
            
            try:
                # Tente de convertir la chaîne de caractères en objet Move
                move = chess.Move.from_uci(move_str)
                
                # Vérifie si le coup a le droit d'être joué
                if move in board.legal_moves:
                    return move
                else:
                    print("❌ Coup illégal dans cette position. Réessaie.")
            except ValueError:
                print("❌ Format invalide. Utilise la notation UCI (ex: e2e4, ou e7e8q pour une promotion).")


class RandomAgent(Base_agent):
    def __init__(self, name="Random_Bot"):
        super().__init__(name)

    def get_move(self, board: chess.Board) -> chess.Move:
        # Récupère tous les coups légaux, les met dans une liste et en tire un au hasard
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves)