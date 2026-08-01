from agents.baseline_agents import HumanAgent
from agents.alphabeta_agent import AlphaBeta_agent
from evaluation.comparison import Game

if __name__ == "__main__":
    print("Initialisation des agents...")
    
    # Tu prends les Blancs, l'Alpha-Beta prend les Noirs (profondeur 3)
    me = HumanAgent(name="Mathéo")
    bot = AlphaBeta_agent(depth=4, name="AlphaBeta_Bot")
    
    # Création de l'arène
    arena = Game(agent_white=me, agent_black=bot)
    
    # Que le meilleur gagne !s
    arena.play_game(display=True)