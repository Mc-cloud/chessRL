import chess
import torch

from agents.base_agent import Base_agent
from utils_NN import CNN, MCTS, UCI_TO_IDX


class NN_Agent(Base_agent):
    """
    Agent qui joue avec le réseau entraîné en self-play (MCTS + CNN).
    Compatible avec le framework Game (evaluation/comparison.py).
    """

    def __init__(self, checkpoint_path, name="NN_Agent", n_simulations=200, device=None):
        super().__init__(name=name)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = CNN(input_channels=13, board_size=8, action_size=len(UCI_TO_IDX))
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.network.load_state_dict(state_dict)
        self.network.to(self.device)
        self.network.eval()

        self.mcts = MCTS(self.network, n_simulations=n_simulations)

    def get_move(self, board: chess.Board) -> chess.Move:
        policy_dict = self.mcts.search(board, add_noise=False)
        best_uci = max(policy_dict, key=policy_dict.get)
        return chess.Move.from_uci(best_uci)