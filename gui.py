import multiprocessing
import pygame
import chess
import sys
import os
from images import *

# Ajustement du chemin pour importer l'agent depuis le dossier courant ou parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.alphabeta_agent import AlphaBeta_agent

sq_size = 600 // 8

IMAGES = {}
def load_images():
    pieces = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
    for piece in pieces:
        img = pygame.image.load(f"images/{piece}.png")
        IMAGES[piece] = pygame.transform.scale(img, (sq_size, sq_size))

def symbole_to_image(piece):
    if piece is None:
        return None
    color = 'w' if piece.color == chess.WHITE else 'b'
    piece_type = piece.symbol().upper()
    return IMAGES[color + piece_type]

def draw_board(screen, board):
    colors = [(238, 238, 210), (118, 150, 86)]
    
    for r in range(8):
        for c in range(8):
            color = colors[(r + c) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(c * sq_size, r * sq_size, sq_size, sq_size))
            
            square = chess.square(c, 7 - r)
            piece = board.piece_at(square)
            
            if piece:
                screen.blit(symbole_to_image(piece), (c * sq_size, r * sq_size))

def ia_worker(queue_requetes, queue_reponses):
    agent = AlphaBeta_agent(depth=4)
    
    while True:
        requete = queue_requetes.get() 
        
        if requete == "QUIT":
            break
            
        board = chess.Board(requete)
        best_move = agent.get_move(board)
        queue_reponses.put(best_move.uci())

def main():
    pygame.init()
    load_images()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("ChessRL - Interface Pygame")
    clock = pygame.time.Clock()
    
    queue_requetes = multiprocessing.Queue()
    queue_reponses = multiprocessing.Queue()
    
    ia_process = multiprocessing.Process(
        target=ia_worker, 
        args=(queue_requetes, queue_reponses)
    )
    ia_process.start()

    board = chess.Board()
    tour_de_ia = False
    running = True
    selected_square = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                queue_requetes.put("QUIT")
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN and not tour_de_ia:
                x, y = pygame.mouse.get_pos()
                col, row = x // (600 // 8), y // (600 // 8)
                clicked_square = chess.square(col, 7 - row)
                
                if selected_square is None:
                    if board.piece_at(clicked_square) and board.color_at(clicked_square) == board.turn:
                        selected_square = clicked_square
                else:
                    move = chess.Move(selected_square, clicked_square)
                    
                    if board.piece_at(selected_square) and board.piece_at(selected_square).piece_type == chess.PAWN:
                        if chess.square_rank(clicked_square) == 0 or chess.square_rank(clicked_square) == 7:
                            move = chess.Move(selected_square, clicked_square, promotion=chess.QUEEN)
                            
                    if move in board.legal_moves:
                        board.push(move)
                        queue_requetes.put(board.fen())
                        tour_de_ia = True
                    
                    selected_square = None

        if tour_de_ia and not queue_reponses.empty():
            coup_ia_uci = queue_reponses.get()
            board.push(chess.Move.from_uci(coup_ia_uci))
            tour_de_ia = False

        draw_board(screen, board)
        
        if tour_de_ia:
            pygame.draw.rect(screen, (255, 0, 0), (0, 0, 600, 10))

        pygame.display.flip()
        clock.tick(60)

    ia_process.join()
    pygame.quit()

if __name__ == "__main__":
    main()