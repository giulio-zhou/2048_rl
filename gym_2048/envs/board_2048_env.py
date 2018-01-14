# Gym-related imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding
# Other imports
import numpy as np

def merge_tiles_in_row(row_vals):
    """
    This method merges the row tiles as if the user attempted to slide them
      from right to left.
    row_vals: An n-dimensional vector of row values. 
    Returns: (1) Row after application of move.
             (2) List of values of merged tiles.
    """
    def _find_next_tile(row_vals, start_index):
        j = start_index
        while j < len(row_vals) and row_vals[j] == 0:
            j += 1
        valid_pos, idx = j < len(row_vals), j
        return (valid_pos, idx)

    i = 0
    current_val = None
    shifted_row_vals = []
    merged_tiles = []
    while i < len(row_vals):
        valid_pos, idx = _find_next_tile(row_vals, i)
        if valid_pos:
            val = row_vals[idx]
            if current_val is None:
                current_val = val
            else:
                # Compare the current_val against the newly found tile.
                if current_val == val:
                    # Merge.
                    merged_tiles.append(2 * val)
                    shifted_row_vals.append(2 * val)
                    current_val = None
                else:
                    shifted_row_vals.append(current_val)
                    current_val = val
            i = idx + 1
        # Handle straggling last tile if no more iterations will occur.
        if not valid_pos or i == len(row_vals):
            if current_val is not None:
                shifted_row_vals.append(current_val)
            break
    new_row_vals = np.zeros(len(row_vals), dtype=np.int32)
    new_row_vals[:len(shifted_row_vals)] = shifted_row_vals
    return new_row_vals, merged_tiles

class Board2048Env(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=np.int32)
        self._generate_random_tile()
        self.discount = 0.95
        # The reward for new appearances of tiles is given by
        #   (tile_value / 2048)^self.reward_scaling
        self.reward_scaling = 2.5

    def _step(self, action):
        if action == 'up':
            board = np.rot90(self.board, k=1)
            reverse_fn = lambda x: np.rot90(x, k=-1)
        elif action == 'down':
            board = np.rot90(self.board, k=-1)
            reverse_fn = lambda x: np.rot90(x, k=1)
        elif action == 'left':
            board = self.board
            reverse_fn = lambda x: x
        elif action == 'right':
            board = np.fliplr(self.board)
            reverse_fn = lambda x: np.fliplr(x)
        else:
            raise Exception("Invalid action \"%s\"" % action)
        new_rows = []
        all_merged_tiles = []
        for row in board:
            new_row, merged_tiles = merge_tiles_in_row(row)
            new_rows.append(new_row)
            all_merged_tiles += merged_tiles
        new_board = reverse_fn(np.array(new_rows))
        if np.array_equal(new_board, self.board):
            # Move left board unchanged.
            return self.board, 0, False, None
        self.board = new_board
        if np.isin(2048, self.board):
            reward = 1
            done = True
        elif np.count_nonzero(self.board) > 0:
            self._generate_random_tile()
            # Compute reward.
            all_merged_tiles = np.array(all_merged_tiles)
            reward = np.power(all_merged_tiles / 2048., self.reward_scaling)
            reward = np.sum(reward)
            done = False
        else:
            reward = -1
            done = True
        return self.board, reward, done, None

    def _reset(self):
        self.__init__()
        return self.board

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(repr(self) + '\n')
        return outfile

    def __repr__(self):
        repr_str = ''
        col_len = 6
        row_border = '-' * (col_len * 4 + 5)
        repr_str += row_border + '\n'
        for row in self.board:
            row_strings = []
            for elem in row: 
                if elem == 0:
                    row_strings.append(' ' * col_len)
                else:
                    elem_str = \
                        str(elem) + ' ' * (col_len - (int(np.log10(elem)) + 1))
                    row_strings.append(elem_str)
            row_str = '|'.join([''] + row_strings + [''])
            repr_str += row_str + '\n'
            repr_str += row_border + '\n'
        return repr_str

    def __str__(self):
        return self.__repr__()

    def _generate_random_tile(self):
        new_val = 2 if np.random.rand() < 0.9 else 4
        ys, xs = np.where(self.board == 0)
        random_idx = np.random.randint(0, len(xs))
        self.board[ys[random_idx], xs[random_idx]] = new_val
