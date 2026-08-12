import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from multiprocessing import get_context, cpu_count
from functools import partial


# ---------------- Vocabulaire des coups ----------------

def create_move_vocab():
    """Crée les dictionnaires de traduction entre les coups UCI et les index du réseau."""
    uci_to_idx = {}
    idx_to_uci = {}
    idx = 0

    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq == to_sq:
                continue

            from_rank = chess.square_rank(from_sq)
            to_rank = chess.square_rank(to_sq)

            is_promotion = (from_rank == 6 and to_rank == 7) or (from_rank == 1 and to_rank == 0)
            file_diff = abs(chess.square_file(from_sq) - chess.square_file(to_sq))

            if is_promotion and file_diff <= 1:
                for promo in ['q', 'r', 'b', 'n']:
                    move_uci = chess.SQUARE_NAMES[from_sq] + chess.SQUARE_NAMES[to_sq] + promo
                    uci_to_idx[move_uci] = idx
                    idx_to_uci[idx] = move_uci
                    idx += 1
            else:
                move_uci = chess.SQUARE_NAMES[from_sq] + chess.SQUARE_NAMES[to_sq]
                uci_to_idx[move_uci] = idx
                idx_to_uci[idx] = move_uci
                idx += 1

    return uci_to_idx, idx_to_uci


UCI_TO_IDX, IDX_TO_UCI = create_move_vocab()


def move_to_index(move: chess.Move):
    return UCI_TO_IDX[move.uci()]


def index_to_move(index: int, board: chess.Board):
    return chess.Move.from_uci(IDX_TO_UCI[index])


# ---------------- Encodage du plateau ----------------

def board_to_tensor(board: chess.Board):
    tensor = np.zeros((13, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)

        channel = piece.piece_type - 1
        if piece.color == chess.BLACK:
            channel += 6

        tensor[channel, rank, file] = 1.0

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    else:
        tensor[12, :, :] = 0.0

    return torch.tensor(tensor)


# ---------------- MCTS ----------------

class Node:
    def __init__(self, state: chess.Board, parent=None, prior_prob=0.0):
        self.state = state
        self.parent = parent
        self.children = {}

        self.n_visits = 0
        self.value_sum = 0
        self.q_value = 0
        self.prior_prob = prior_prob

    def expand(self, action_probs):
        for move, prob in action_probs.items():
            if move not in self.children:
                next_state = self.state.copy(stack=False)
                next_state.push(chess.Move.from_uci(move))
                self.children[move] = Node(state=next_state, parent=self, prior_prob=prob)

    def is_expended(self):
        return len(self.children) > 0

    def best_child(self, c):
        best_score = -math.inf
        best_action = None
        best_child = None

        for action, child in self.children.items():
            q_val = child.q_value
            u_val = c * child.prior_prob * math.sqrt(self.n_visits) / (1 + child.n_visits)
            puct_score = q_val + u_val

            if puct_score > best_score:
                best_score = puct_score
                best_action = action
                best_child = child

        return best_action, best_child

    def backpropagate(self, value):
        # Iterative instead of recursive: avoids Python function-call overhead
        # (this runs once per simulation, so it adds up fast at 800 sims/move).
        node = self
        v = value
        while node is not None:
            node.n_visits += 1
            node.value_sum += v
            node.q_value = node.value_sum / node.n_visits
            node = node.parent
            v = -v


class MCTS:
    def __init__(self, neural_net, c=1.5, n_simulations=800):
        self.nn = neural_net
        self.c = c
        self.n_simulations = n_simulations

    def search(self, initial_state: chess.Board):
        root = Node(state=initial_state)

        for _ in range(self.n_simulations):
            node = root

            while node.is_expended():
                action, node = node.best_child(self.c)

            if node.state.is_game_over():
                value = -1.0 if node.state.is_checkmate() else 0.0
            else:
                action_probs, value = self.nn.predict(node.state)

                legal_moves = [m.uci() for m in node.state.legal_moves]
                legal_probs = {m: prob for m, prob in action_probs.items() if m in legal_moves}

                sum_probs = sum(legal_probs.values())
                if sum_probs > 0:
                    legal_probs = {m: prob / sum_probs for m, prob in legal_probs.items()}
                else:
                    legal_probs = {m: 1.0 / len(legal_moves) for m in legal_moves}

                node.expand(legal_probs)

            node.backpropagate(-value)

        action_visits = {action: child.n_visits for action, child in root.children.items()}
        sum_visits = sum(action_visits.values())
        mcts_policy = {action: visits / sum_visits for action, visits in action_visits.items()}

        return mcts_policy


# ---------------- Réseau de neurones ----------------

class CNN(nn.Module):
    def __init__(self, input_channels, board_size, action_size, hidden_dim=64):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size

        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.bn4 = nn.BatchNorm2d(hidden_dim)

        self.policy_conv = nn.Conv2d(hidden_dim, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)

        self.value_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        policy_out = F.log_softmax(p, dim=1)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        value_out = torch.tanh(v)

        return policy_out, value_out

    def predict(self, board):
        # Note: pas de self.eval() ici - appeler ça à chaque simulation
        # (800x par coup) ajoute un vrai coût de bascule des BatchNorm.
        # Le net doit déjà être en mode eval avant l'appel (voir play_one_game).
        board_tensor = board_to_tensor(board)
        device = next(self.parameters()).device
        board_tensor = board_tensor.to(device)

        with torch.no_grad():
            board_tensor = board_tensor.unsqueeze(0)
            log_policy, value = self.forward(board_tensor)
            policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
            value = value.item()

        policy_dict = {}
        for i in range(len(policy)):
            if i in IDX_TO_UCI:
                move_uci = IDX_TO_UCI[i]
                policy_dict[move_uci] = policy[i]

        return policy_dict, value


# ---------------- Self-play ----------------

def play_one_game(neural_net, num_simulations=800, temp_threshold=15):
    board = chess.Board()
    memory = []
    move_count = 0

    neural_net.eval()
    mcts = MCTS(neural_net, n_simulations=num_simulations)

    while not board.is_game_over():
        move_count += 1
        policy_dict = mcts.search(board)
        action_size = len(UCI_TO_IDX)
        policy_vector = np.zeros(action_size, dtype=np.float32)

        for uci_move, prob in policy_dict.items():
            idx = UCI_TO_IDX[uci_move]
            policy_vector[idx] = prob

        memory.append((board_to_tensor(board), policy_vector, board.turn))
        actions = list(policy_dict.keys())
        probs = list(policy_dict.values())

        if move_count <= temp_threshold:
            chosen_uci = np.random.choice(actions, p=probs)
        else:
            chosen_uci = actions[np.argmax(probs)]

        board.push(chess.Move.from_uci(chosen_uci))

    result = board.result()
    if result == "1-0":
        winner = chess.WHITE
    elif result == "0-1":
        winner = chess.BLACK
    else:
        winner = None

    dataset = []
    for state_tensor, policy_vector, player_turn in memory:
        if winner is None:
            reward = 0.0
        elif player_turn == winner:
            reward = 1.0
        else:
            reward = -1.0

        dataset.append((state_tensor, policy_vector, reward))

    return dataset


def _self_play_worker(_, neural_net, num_simulations, temp_threshold):
    return play_one_game(
        neural_net,
        num_simulations=num_simulations,
        temp_threshold=temp_threshold
    )


def generate_games_with_self_play(neural_net, num_games=10, num_simulations=100,
                                   temp_threshold=15, n_workers=None):
    """
    Joue plusieurs parties d'affilée contre soi-même et fusionne toutes
    les positions générées dans un seul grand dataset d'entraînement.

    Utilise multiprocessing avec la méthode "spawn" (fiable dans un notebook
    Colab/Jupyter, contrairement à "fork" qui peut deadlock silencieusement).
    """
    dataset_total = []

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    # Les workers font de l'inférence CPU (petit réseau, positions une par
    # une - le GPU n'aide ici qu'avec du batching, ce que cette boucle ne
    # fait pas). Le réseau pickle vers les workers doit donc être sur CPU.
    neural_net.to("cpu")
    neural_net.eval()

    print(f"\n🔄 Début de la génération de {num_games} parties en Self-Play "
          f"({n_workers} workers en parallèle, méthode spawn)...")

    worker_fn = partial(
        _self_play_worker,
        neural_net=neural_net,
        num_simulations=num_simulations,
        temp_threshold=temp_threshold
    )

    ctx = get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for i, game_data in enumerate(pool.imap_unordered(worker_fn, range(num_games))):
            dataset_total.extend(game_data)
            print(f"♟️ Partie {i+1}/{num_games} terminée...")

    print(f"✅ Génération terminée ! {len(dataset_total)} positions collectées.")

    return dataset_total