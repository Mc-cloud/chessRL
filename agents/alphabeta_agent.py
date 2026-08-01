import chess
import math
import sys

from .base_agent import Base_agent

class AlphaBeta_agent(Base_agent):
    def __init__(self, depth = 3, name = "AlphaBeta"):
        super().__init__(name)

        self.depth = depth
        self.piece_values = {
            chess.PAWN : 100,
            chess.BISHOP : 330,
            chess.ROOK : 500,
            chess.KNIGHT : 320,
            chess.QUEEN : 900,
            chess.KING : 0
        }

#https://www.chessprogramming.org/Simplified_Evaluation_Function

        self.pst = {
            chess.PAWN: [
                 0,  0,  0,  0,  0,  0,  0,  0,
                 5, 10, 10,-20,-20, 10, 10,  5,
                 5, -5,-10,  0,  0,-10, -5,  5,
                 0,  0,  0, 20, 20,  0,  0,  0,
                 5,  5, 10, 25, 25, 10,  5,  5,
                10, 10, 20, 30, 30, 20, 10, 10,
                50, 50, 50, 50, 50, 50, 50, 50,
                 0,  0,  0,  0,  0,  0,  0,  0
            ],
            chess.KNIGHT: [
                -50,-40,-30,-30,-30,-30,-40,-50,
                -40,-20,  0,  5,  5,  0,-20,-40,
                -30,  5, 10, 15, 15, 10,  5,-30,
                -30,  0, 15, 20, 20, 15,  0,-30,
                -30,  5, 15, 20, 20, 15,  5,-30,
                -30,  0, 10, 15, 15, 10,  0,-30,
                -40,-20,  0,  0,  0,  0,-20,-40,
                -50,-40,-30,-30,-30,-30,-40,-50
            ],
            chess.BISHOP: [
                -20,-10,-10,-10,-10,-10,-10,-20,
                -10,  5,  0,  0,  0,  0,  5,-10,
                -10, 10, 10, 10, 10, 10, 10,-10,
                -10,  0, 10, 10, 10, 10,  0,-10,
                -10,  5,  5, 10, 10,  5,  5,-10,
                -10,  0,  5, 10, 10,  5,  0,-10,
                -10,  0,  0,  0,  0,  0,  0,-10,
                -20,-10,-10,-10,-10,-10,-10,-20
            ],
            chess.ROOK: [
                  0,  0,  0,  5,  5,  0,  0,  0,
                 -5,  0,  0,  0,  0,  0,  0, -5,
                 -5,  0,  0,  0,  0,  0,  0, -5,
                 -5,  0,  0,  0,  0,  0,  0, -5,
                 -5,  0,  0,  0,  0,  0,  0, -5,
                 -5,  0,  0,  0,  0,  0,  0, -5,
                  5, 10, 10, 10, 10, 10, 10,  5,
                  0,  0,  0,  0,  0,  0,  0,  0
            ],
            chess.QUEEN: [
                -20,-10,-10, -5, -5,-10,-10,-20,
                -10,  0,  5,  0,  0,  0,  0,-10,
                -10,  5,  5,  5,  5,  5,  0,-10,
                  0,  0,  5,  5,  5,  5,  0, -5,
                 -5,  0,  5,  5,  5,  5,  0, -5,
                -10,  0,  5,  5,  5,  5,  0,-10,
                -10,  0,  0,  0,  0,  0,  0,-10,
                -20,-10,-10, -5, -5,-10,-10,-20
            ],
            chess.KING: [
                 20, 30, 10,  0,  0, 10, 30, 20,
                 20, 20,  0,  0,  0,  0, 20, 20,
                -10,-20,-20,-20,-20,-20,-20,-10,
                -20,-30,-30,-40,-40,-30,-30,-20,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -50,-40,-30,-20,-20,-30,-40,-50
            ]
        }

    def _evaluate_board(self, board : chess.Board):
        if board.is_checkmate():
            return -99999 if board.turn else 99999

        if board.is_game_over():
            return 0

        score = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                val = self.piece_values[piece.piece_type]

                table = self.pst[piece.piece_type]

                #on lit dans l'autre sens si on a les noirs
                idx = square if piece.color == chess.WHITE else chess.square_mirror(square)

                pst_val = table[idx]

                if piece.color == chess.WHITE:
                    score += val + pst_val

                else :
                    score -= val + pst_val
        return score
    
    #pb rencontré : même en augmentant la profondeur, l'algo peut s'arrêter juste avant la fin d'une tactique (ex : il capture un pion avec sa dame mais le fou adverse vient la capturer après)
    # ==> Ajout de fonction pour évaluer si il y a encore des captures possibles. Si oui, on continue de calculer, sinon on peut arrêter
    def _quiescence(self, board : chess.Board, alpha, beta, maximizing_player):
        stand_eval = self._evaluate_board(board)

        if maximizing_player:
            if stand_eval >= beta:
                return beta
            alpha = max(alpha, stand_eval)

        else : 
            if stand_eval <= alpha:
                return alpha
            beta = min(beta, stand_eval)

        for move in board.legal_moves:
            if not board.is_capture(move):
                continue
            board.push(move)
            score = self._quiescence(board, alpha, beta, not maximizing_player)
            board.pop()

            if maximizing_player:
                if score > alpha :
                    alpha = score
                if alpha >= beta :
                    return beta

            else : 
                if score < beta :
                    beta = score

                if beta <= alpha:
                    return alpha

        return alpha if maximizing_player else beta


    #pour qu'ils considèrent les captures et les coups qui semblent être les meilleurs avant comme ça il élague les lignes plus rapidement
    def _order_moves(self, board, moves):
        def score(move):
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                v_val = self.piece_values[victim.piece_type] if victim else 100  # en passant
                a_val = self.piece_values[attacker.piece_type] if attacker else 0
                return 10000 + v_val - a_val
            return 0
        return sorted(moves, key=score, reverse=True)

    def _alphabeta(self, board : chess.Board, depth : int, alpha : float, beta : float, maximizing_player : bool):
        if depth == 0 or board.is_game_over():
            return self._quiescence(board, alpha, beta, maximizing_player)

        if maximizing_player:
            max_eval = -math.inf
            for move in self._order_moves(board, board.legal_moves):
                board.push(move)
                eval_score = self._alphabeta(board, depth - 1, alpha, beta, False)
                board.pop()

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                if beta <= alpha :
                    break

            return max_eval

        else :
            min_eval = math.inf
            for move in self._order_moves(board, board.legal_moves):
                board.push(move)
                eval_score = self._alphabeta(board, depth - 1, alpha, beta, True)
                board.pop()

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                if beta <= alpha:
                    break

            return min_eval


    def get_move(self, board : chess.Board):
        best_move = None

        is_maximising = (board.turn == chess.WHITE)

        if is_maximising:
            best_score = - math.inf
            for move in self._order_moves(board, board.legal_moves):
                board.push(move)
                score = self._alphabeta(board, self.depth - 1, -math.inf, math.inf, False)
                board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move

        else : 
            best_score = math.inf
            for move in self._order_moves(board, board.legal_moves):
                board.push(move)
                score = self._alphabeta(board, self.depth - 1, -math.inf, math.inf, True)
                board.pop()
                
                if score < best_score:
                    best_score = score
                    best_move = move

        formatted_score = f"{best_score / 100:+.2f}"
        print(f"📊 Evaluation: {formatted_score}", file = sys.stderr)

        return best_move