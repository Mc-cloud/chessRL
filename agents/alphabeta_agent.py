import chess
import chess.polyglot
import math
import sys
import time

class TimeoutException(Exception):
    pass

from .base_agent import Base_agent

EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


class AlphaBeta_agent(Base_agent):
    def __init__(self, depth = 99, name = "AlphaBeta", time_limit = 5.0):
        super().__init__(name)

        self.depth = depth
        self.time_limit = time_limit
        self.piece_values_mg = {
            chess.PAWN : 124,
            chess.BISHOP : 825,
            chess.ROOK : 1276,
            chess.KNIGHT : 781,
            chess.QUEEN : 2538,
            chess.KING : 0
        }

        self.piece_values_eg = {
            chess.PAWN : 206,
            chess.BISHOP : 915,
            chess.ROOK : 1380,
            chess.KNIGHT : 854,
            chess.QUEEN : 2682,
            chess.KING : 0
        }

        self.phase_weights = {
            chess.PAWN : 0,
            chess.KNIGHT: 1,
            chess.BISHOP : 1,
            chess.ROOK : 2,
            chess.QUEEN : 4,
            chess.KING : 0
        }

        self.max_phase = 24

#https://hxim.github.io/Stockfish-Evaluation-Guide/

        self.pst_mg = {
            chess.PAWN: [
                 0,  0,  0,  0,  0,  0,  0,  0,
                 3,  3, 10, 19, 16, 19,  7, -5,
                -9,-15, 11, 15, 32, 22,  5,-22,
                13,  0,-13,  1, 11, -2,-13,  5,
                -4,-23,  6, 20, 40, 17,  4, -8,
                 5,-12, -7, 22, -8, -5,-15, -8,
                -7,  7, -3,-13,  5,-16, 10, -8,
                 0,  0,  0,  0,  0,  0,  0,  0
            ],
            chess.KNIGHT: [
                -175,-92,-74,-73,-73,-74,-92,-175,
                 -77,-41,-27,-15,-15,-27,-41, -77,
                 -61,-17,  6, 12, 12,  6,-17, -61,
                 -35,  8, 40, 49, 49, 40,  8, -35,
                 -34, 13, 44, 51, 51, 44, 13, -34,
                  -9, 22, 58, 53, 53, 58, 22,  -9,
                 -67,-27,  4, 37, 37,  4,-27, -67,
                -201,-83,-56,-26,-26,-56,-83,-201
            ],
            chess.BISHOP: [
                -53, -5, -8,-23,-23, -8, -5,-53,
                -15,  8, 19,  4,  4, 19,  8,-15,
                 -7, 21, -5, 17, 17, -5, 21, -7,
                 -5, 11, 25, 39, 39, 25, 11, -5,
                -12, 29, 22, 31, 31, 22, 29,-12,
                -16,  6,  1, 11, 11,  1,  6,-16,
                -17,-14,  5,  0,  0,  5,-14,-17,
                -48,  1,-14,-23,-23,-14,  1,-48
            ],
            chess.ROOK: [
                -31,-20,-14, -5, -5,-14,-20,-31,
                -21,-13, -8,  6,  6, -8,-13,-21,
                -25,-11, -1,  3,  3, -1,-11,-25,
                -13, -5, -4, -6, -6, -4, -5,-13,
                -27,-15, -4,  3,  3, -4,-15,-27,
                -22, -2,  6, 12, 12,  6, -2,-22,
                 -2, 12, 16, 18, 18, 16, 12, -2,
                -17,-19, -1,  9,  9, -1,-19,-17
            ],
            chess.QUEEN: [
                 3, -5, -5,  4,  4, -5, -5,  3,
                -3,  5,  8, 12, 12,  8,  5, -3,
                -3,  6, 13,  7,  7, 13,  6, -3,
                 4,  5,  9,  8,  8,  9,  5,  4,
                 0, 14, 12,  5,  5, 12, 14,  0,
                -4, 10,  6,  8,  8,  6, 10, -4,
                -5,  6, 10,  8,  8, 10,  6, -5,
                -2, -2,  1, -2, -2,  1, -2, -2
            ],
            chess.KING: [
                271,327,271,198,198,271,327,271,
                278,303,234,179,179,234,303,278,
                195,258,169,120,120,169,258,195,
                164,190,138, 98, 98,138,190,164,
                154,179,105, 70, 70,105,179,154,
                123,145, 81, 31, 31, 81,145,123,
                 88,120, 65, 33, 33, 65,120, 88,
                 59, 89, 45, -1, -1, 45, 89, 59
            ]
        }

        self.pst_eg = {
            chess.PAWN : [
                  0,  0,  0,  0,  0,  0,  0,  0,
                -10, -6, 10,  0, 14,  7, -5,-19,
                -10,-10,-10,  4,  4,  3, -6, -4,  
                  6, -2, -8, -4,-13,-12,-10, -9,
                 10,  5,  4, -5, -5, -5, 14,  9,
                 28, 20, 21, 28, 30,  7,  6, 13,
                  0,-11, 12, 21, 25, 19,  4,  7, 
                  0,  0,  0,  0,  0,  0,  0,  0
            ],
            chess.KNIGHT : [
                -96,-65,-49,-21,-21,-49,-65,-96,
                -67,-54,-18,  8,  8,-18,-54,-67,
                -40,-27, -8, 29, 29, -8,-27,-40,
                -35, -2, 13, 28, 28, 13, -2,-35,
                -45,-16,  9, 39, 39,  9,-16,-45,
                -51,-44,-16, 17, 17,-16,-44,-51,
                -69,-50,-51, 12, 12,-51,-50,-69,
               -100,-88,-56,-17,-17,-56,-88,-100
            ],
            chess.BISHOP : [
                -57,-30,-37,-12,-12,-37,-30,-57,
                -37,-13,-17,  1,  1,-17,-13,-37,
                -16, -1, -2, 10, 10, -2, -1,-16,
                -20, -6,  0, 17, 17,  0, -6,-20,
                -17, -1,-14, 15, 15,-14, -1,-17,
                -30,  6,  4,  6,  6,  4,  6,-30,
                -31,-20, -1,  1,  1, -1,-20,-31,
                -46,-42,-37,-24,-24,-37,-42,-46
            ],
            chess.ROOK : [
                 -9,-13,-10, -9, -9,-10,-13, -9,
                -12, -9, -1, -2, -2, -1, -9,-12,
                  6, -8, -2, -6, -6, -2, -8,  6,
                 -6,  1, -9,  7,  7, -9,  1, -6,
                 -5,  8,  7, -6, -6,  7,  8, -5,
                  6,  1, -7, 10, 10, -7,  1, -6,
                  4,  5, 20, -5, -5, 20,  5,  4,
                 18,  0, 19, 13, 13, 19,  0, 18
            ],

            chess.QUEEN : [
                -69, -57, -47, -26, -26, -47, -57, -69, 
                -55, -31, -22,  -4,  -4, -22, -31, -55,
                -39, -18,  -9,   3,   3,  -9, -18, -39,
                -23,  -3,  13,  24,  24,  13,  -3, -23,
                -29,  -6,   9,  21,  21,   9,  -6, -29,
                -38, -18, -12,   1,   1, -12, -18, -38,
                -50, -27, -24,  -8,  -8, -24, -27, -50,
                -75, -52, -43, -36, -36, -43, -52, -75 
            ],

            chess.KING : [
                  1,  45,  85,  76,  76,  85,  45,   1, 
                 53, 100, 133, 135, 135, 133, 100,  53, 
                 88, 130, 169, 175, 175, 169, 130,  88, 
                103, 156, 172, 172, 172, 172, 156, 103, 
                 96, 166, 199, 199, 199, 199, 166,  96, 
                 92, 172, 184, 191, 191, 184, 172,  92, 
                 47, 121, 116, 131, 131, 116, 121,  47, 
                 11,  59,  73,  78,  78,  73,  59,  11  
            ]
        }

        self.mobility_mg = {
            chess.KNIGHT : [-62,-53,-12,-4,3,13,22,28,33],
            chess.BISHOP : [-48,-20,16,26,38,51,55,63,63,68,81,81,91,98],
            chess.ROOK : [-60,-20,2,3,3,11,22,31,40,40,41,48,57,57,62],
            chess.QUEEN : [-30,-12,-8,-9,20,23,23,35,38,53,64,65,65,66,67,67,72,72,77,79,93,108,108,108,110,114,114,116]
        }

        self.mobility_eg = {
            chess.KNIGHT :  [-81,-56,-31,-16,5,11,17,20,25],
            chess.BISHOP : [-59,-23,-3,13,24,42,54,57,65,73,78,86,88,97],
            chess.ROOK : [-78,-17,23,39,70,99,103,121,134,139,158,164,168,169,172],
            chess.QUEEN : [-48,-30,-7,19,40,55,59,75,78,96,96,100,121,127,131,133,136,141,147,150,151,168,168,171,182,182,192,219]
        }

        self.transposition_table = {}

    #pour éviter les zugzwang dans le null-move pruning(voir alphabeta), 
    # on regarde si il reste des pièces 
    #(il y a toujours bcp de cas où il y aura de zugzwang, à voir 
    # si c'est intéressant de le garder)
    def _has_non_pawn_material(self, board : chess.Board, color : bool):
        return bool(
            board.pieces_mask(chess.KNIGHT, color) |
            board.pieces_mask(chess.BISHOP, color) |
            board.pieces_mask(chess.ROOK, color) |
            board.pieces_mask(chess.QUEEN, color)
        )

    def _evaluate_board(self, board : chess.Board):
        if board.is_checkmate():
            return -99999 if board.turn else 99999

        if board.is_game_over():
            return 0

        mg_score = 0
        eg_score = 0
        phase = 0


        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                val_mg = self.piece_values_mg[piece.piece_type]
                val_eg = self.piece_values_eg[piece.piece_type]

                phase += self.phase_weights[piece.piece_type]

                #on lit dans l'autre sens si on a les noirs
                idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
                mg_pst = self.pst_mg[piece.piece_type][idx]
                eg_pst = self.pst_eg[piece.piece_type][idx]

                mob_mg = 0
                mob_eg = 0
                struct_mg = 0
                struct_eg = 0

                if piece.piece_type in self.mobility_mg:
                    nb_moves = len(board.attacks(square))

                    max_idx_mg = len(self.mobility_mg[piece.piece_type]) - 1
                    max_idx_eg = len(self.mobility_eg[piece.piece_type]) - 1

                    mob_mg = self.mobility_mg[piece.piece_type][min(nb_moves, max_idx_mg)]
                    mob_eg = self.mobility_eg[piece.piece_type][min(nb_moves, max_idx_eg)]

                if piece.piece_type == chess.PAWN :
                    file = chess.square_file(square)
                    rank = chess.square_rank(square)

                    file_mask = chess.BB_FILES[file]
                    adj_mask = 0
                    if file > 0: adj_mask |= chess.BB_FILES[file - 1]
                    if file < 7: adj_mask |= chess.BB_FILES[file + 1]

                    friendly_pawns = board.pieces_mask(chess.PAWN, piece.color)
                    enemy_pawns = board.pieces_mask(chess.PAWN, not piece.color)
                    
                    # pions doublés
                    if (friendly_pawns & file_mask).bit_count() > 1:
                        struct_mg -= 11
                        struct_eg -= 11
                        
                    # pions isolés
                    if not (friendly_pawns & adj_mask):
                        struct_mg -= 5
                        struct_eg -= 15
                        
                    # pions passés
                    front_mask = 0
                    if piece.color == chess.WHITE:
                        for r in range(rank + 1, 8): front_mask |= chess.BB_RANKS[r]
                    else:
                        for r in range(0, rank): front_mask |= chess.BB_RANKS[r]
                        
                    if not (enemy_pawns & (file_mask | adj_mask) & front_mask):
                        struct_mg += 20
                        struct_eg += 40


                if piece.color == chess.WHITE:
                    mg_score += val_mg + mg_pst + mob_mg + struct_mg
                    eg_score += val_eg + eg_pst + mob_eg + struct_eg
                else :
                    mg_score -= val_mg + mg_pst + mob_mg + struct_mg
                    eg_score -= val_eg + eg_pst + mob_eg + struct_eg

        if phase > 16:
            # 1. Comptage rapide des pions bloqués (un pion blanc avec un pion noir juste au-dessus)
            white_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
            black_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)
            
            blocked_pawns = (white_pawns << 8) & black_pawns
            blocked_count = blocked_pawns.bit_count() * 2

            piece_count = len(board.piece_map())
            
            weight = piece_count - 3 + min(blocked_count, 9)
        
            white_space_mask = 0x000000003C3C3C00
            black_space_mask = 0x003C3C3C00000000

            black_pawn_attacks = ((black_pawns >> 7) & ~chess.BB_FILE_A) | ((black_pawns >> 9) & ~chess.BB_FILE_H)
            white_pawn_attacks = ((white_pawns << 7) & ~chess.BB_FILE_H) | ((white_pawns << 9) & ~chess.BB_FILE_A)
            
            white_safe_space = white_space_mask & ~white_pawns & ~black_pawn_attacks
            black_safe_space = black_space_mask & ~black_pawns & ~white_pawn_attacks
            
            white_space_area = white_safe_space.bit_count()
            black_space_area = black_safe_space.bit_count()

            white_space_bonus = int((white_space_area * weight * weight) / 16)
            black_space_bonus = int((black_space_area * weight * weight) / 16)
            
            mg_score += white_space_bonus
            mg_score -= black_space_bonus

        phase = min(phase, self.max_phase)

        tapered_score = (mg_score * phase + eg_score *(self.max_phase - phase))/self.max_phase

        return tapered_score
    
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

        legal_moves = self._order_moves(board, list(board.legal_moves))

        for move in legal_moves:
            is_capture = board.is_capture(move)
            is_check = board.gives_check(move)

            if not is_capture and not is_check:
                continue

            piece_move = board.piece_at(move.from_square)

            board.push(move)

            is_safe = True

            if is_check :
                for square in board.attackers(board.turn, move.to_square):
                    attacker = board.piece_at(square)
                    if attacker is not None :
                        if not is_capture:
                            is_safe = False
                            break

                        else :
                            if self.piece_values_mg[attacker.piece_type] <= self.piece_values_mg[piece_move.piece_type]:
                                is_safe = False
                                break

            if not is_safe:
                board.pop()
                continue
                # sinon, on garde le push actuel, pas besoin de repush plus bas


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


    #pour qu'ils considèrent les captures et les coups qui semblent être les meilleurs avant comme ça il élague plus de coups
    def _order_moves(self, board, moves, tt_move=None):
        def score(move):
            if tt_move is not None and move == tt_move:
                return 1000000
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                v_val = self.piece_values_mg[victim.piece_type] if victim else 100  # en passant
                a_val = self.piece_values_mg[attacker.piece_type] if attacker else 0
                return 10000 + v_val - a_val
            return 0
        return sorted(moves, key=score, reverse=True)

    def _alphabeta(self, board : chess.Board, depth : int, alpha : float, beta : float, maximizing_player : bool):
        self.nodes += 1
        if self.nodes % 2048 == 0 :
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutException()
        alpha_orig = alpha
        beta_orig = beta

        key = chess.polyglot.zobrist_hash(board)
        tt_entry = self.transposition_table.get(key)
        tt_move = None

        if tt_entry is not None and tt_entry["depth"] >= depth :
            tt_move = tt_entry.get("best_move")
            flag = tt_entry["flag"]
            value = tt_entry["value"]

            if flag == EXACT:
                return value
            elif flag == LOWERBOUND :
                alpha = max(alpha, value)
            elif flag == UPPERBOUND :
                beta = min(beta, value)
            if alpha >= beta:
                return value

        elif tt_entry is not None :
            tt_move = tt_entry.get("best_move")

        #l'idée c'est de se demander si on passe son tour, est ce que la position est toujours bonne
        # si elle l'est alors ça nous permet d'élaguer plus tôt
        R = 3
        NULL_MOVE_MIN_DEPTH = 3
        if (depth >= NULL_MOVE_MIN_DEPTH and not board.is_check() and self._has_non_pawn_material(board, board.turn) and beta < math.inf):
            board.push(chess.Move.null())
            null_score = self._alphabeta(board, depth - 1 - R,alpha, beta, not maximizing_player)
            board.pop()

            if maximizing_player and null_score >= beta:
                return beta
            if not maximizing_player and null_score <= alpha:
                return alpha
        
        if depth <= 0 or board.is_game_over():
            return self._quiescence(board, alpha, beta, maximizing_player)

        best_move_found = None
        legal_moves = self._order_moves(board, board.legal_moves)

        if maximizing_player:
            max_eval = -math.inf
            for move in legal_moves:
                board.push(move)
                eval_score = self._alphabeta(board, depth - 1, alpha, beta, False)
                board.pop()

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move_found = move
                    
                alpha = max(alpha, eval_score)

                if beta <= alpha :
                    break

            result = max_eval

        else :
            min_eval = math.inf
            for move in legal_moves:
                board.push(move)
                eval_score = self._alphabeta(board, depth - 1, alpha, beta, True)
                board.pop()

                if eval_score < min_eval :
                    min_eval = eval_score
                    best_move_found = move

                beta = min(beta, eval_score)

                if beta <= alpha:
                    break
            result = min_eval

        if result <= alpha_orig:
            flag = UPPERBOUND
        elif result >= beta_orig:
            flag = LOWERBOUND
        else:
            flag = EXACT

        self.transposition_table[key] = {
                "value":result,
                "depth":depth,
                "flag":flag,
                "best_move" : best_move_found
            }
        return result


    # Ajout d'une détection de mat en 1 car l'algorithme ne préfère pas forcément le mat en 1 à un mat en 2, les deux ayant 999 en valeur.
    def _checkmate_in_one(self, board : chess.Board):
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            board.pop()

    
    def get_move(self, board : chess.Board, time_left : float = None, increment : float = 0.0):
        if time_left is not None :
            self.time_limit = (time_left / 40.0) + (increment / 2.0)
            self.time_limit = min(self.time_limit, max(0.1, time_left - 0.5))
        else :
            self.time_limit = 10.0

        self.start_time = time.time()
        self.nodes = 0

        best_move_overall = None
        best_score_overall = 0 
        is_maximising = (board.turn == chess.WHITE)

        original_stack_len = len(board.move_stack)

        try :
            for current_depth in range(1, self.depth + 1):
                best_move_depth = None
                alpha = -math.inf
                beta = math.inf
                
                key = chess.polyglot.zobrist_hash(board)
                tt_entry = self.transposition_table.get(key)
                tt_move = tt_entry.get("best_move") if tt_entry else None
                if best_move_overall :
                    tt_move = best_move_overall

                legal_moves = self._order_moves(board, list(board.legal_moves), tt_move=tt_move)

                if is_maximising:
                    best_score = - math.inf
                    
                    for move in legal_moves:
                        board.push(move)
                        score = self._alphabeta(board, current_depth - 1, alpha, beta, False)
                        board.pop()

                        if score > best_score:
                            best_score = score
                            best_move_depth = move

                        alpha = max(alpha, best_score)

                else : 
                    best_score = math.inf
                    for move in legal_moves:
                        board.push(move)
                        score = self._alphabeta(board, current_depth - 1, alpha, beta, True)
                        board.pop()
                        
                        if score < best_score:
                            best_score = score
                            best_move_depth = move

                        beta = min(beta, best_score)

                best_move_overall = best_move_depth    
                best_score_overall = best_score
                formatted_score = f"{best_score_overall / 100:+.2f}"
                elapsed_time = time.time() - self.start_time
                print(f"✅ [Profondeur {current_depth}] terminée en {elapsed_time:.2f}s | Éval: {formatted_score} | Coup: {best_move_overall}", file=sys.stderr)
        except TimeoutException:
            print(f'{self.time_limit} depasse', file = sys.stderr)
            while len(board.move_stack) > original_stack_len :
                board.pop()
        if best_move_overall is None :
            best_move_overall = list(board.legal_moves)[0]

        #Ajout d'une vérification de mat en 1 pour s'assurer que l'IA ne rate pas un mat immédiat
        checkmate_move = self._checkmate_in_one(board)
        if checkmate_move is not None:
            print("✅ Mat en 1 détecté !", file=sys.stderr)
            best_move_overall = checkmate_move

        return best_move_overall