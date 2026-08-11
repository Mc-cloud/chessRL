import sys
import chess

from agents.alphabeta_agent import AlphaBeta_agent

DEPTH = 99

agent = AlphaBeta_agent(depth=DEPTH)
board = chess.Board()

def send(msg: str):
    print(msg, flush=True)

def handle_position(parts):
    global board

    if "startpos" in parts:
        board = chess.Board()
        if "moves" in parts:
            idx = parts.index("moves") + 1
            for move_str in parts[idx:]:
                board.push_uci(move_str)

    elif "fen" in parts:
        fen_idx = parts.index("fen") + 1
        fen = " ".join(parts[fen_idx:fen_idx + 6])
        board = chess.Board(fen)
        if "moves" in parts:
            idx = parts.index("moves") + 1
            for move_str in parts[idx:]:
                board.push_uci(move_str)

def main():
    global board

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        cmd = parts[0]

        if cmd == "uci":
            send("id name AlphaBetaBot")
            send("id author Mathe")
            send("option name Move Overhead type spin default 100 min 0 max 5000")
            send("option name Threads type spin default 1 min 1 max 512")
            send("option name Hash type spin default 16 min 1 max 1024")
            send("option name MultiPV type spin default 1 min 1 max 500")
            send("option name Ponder type check default false")
            send("option name SyzygyPath type string default <empty>")
            send("option name SyzygyProbeLimit type spin default 7 min 0 max 7")
            send("option name UCI_Chess960 type check default false")
            send("option name UCI_Variant type combo default chess var chess")
            send("option name UCI_ShowWDL type check default false")
            send("option name Skill Level type spin default 20 min 0 max 20")
            send("option name UCI_LimitStrength type check default false")
            send("option name UCI_Elo type spin default 1350 min 1350 max 2850")
            send("option name Slow Mover type spin default 100 min 10 max 1000")
            send("option name Contempt type spin default 0 min -100 max 100")
            send("uciok")

        elif cmd == "isready":
            send("readyok")

        elif cmd == "ucinewgame":
            board = chess.Board()
            agent.transposition_table.clear()

        elif cmd == "setoption":
            pass  # ignoré, notre moteur n'a pas besoin de ces réglages

        elif cmd == "position":
            handle_position(parts)

        elif cmd == "go":
            wtime = btime = winc = binc = 0.0

            if "wtime" in parts :
                wtime = int(parts[parts.index("wtime") + 1]) / 1000.0
            if "btime" in parts:
                btime = int(parts[parts.index("btime") + 1]) / 1000.0
            if "winc" in parts:
                winc = int(parts[parts.index("winc") + 1]) / 1000.0
            if "binc" in parts:
                binc = int(parts[parts.index("binc") + 1]) / 1000.0

            if board.turn == chess.WHITE:
                time_left = wtime if wtime > 0 else None
                increment = winc
            else:
                time_left = btime if btime > 0 else None
                increment = binc

            move = agent.get_move(board, time_left = time_left, increment = increment)
            send(f"bestmove {move.uci()}")

        elif cmd == "quit":
            break

if __name__ == "__main__":
    main()