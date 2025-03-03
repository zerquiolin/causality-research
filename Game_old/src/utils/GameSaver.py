
import pickle

class GameSaver:
    def save_game(self, game, filename='game_state.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(game, f)

    def load_game(self, filename='game_state.pkl'):
        with open(filename, 'rb') as f:
            return pickle.load(f)
