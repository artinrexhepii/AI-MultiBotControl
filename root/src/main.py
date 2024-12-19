import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.game import Game

if __name__ == "__main__":
    game = Game()
    game.run() 