"""
This module contains class implementations for three environments:
- Rubik's Cube: Cube3

Please note that we prioritize readability and reproducibility over speed optimization in this repository.
"""

import random
import numpy as np

class Cube3:
    """
    A class for 3x3x3 Rubik's Cube
    """
    def __init__(self):
        self.DTYPE = np.int64

        # Define initial and goal state
        self.reset()
        self.goal = np.arange(0, 9 * 6, dtype=self.DTYPE) // 9

        # Define moves
        ## faces and turns
        faces = ["U", "D", "L", "R", "B", "F"]
        ## [90 degrees clockwise, 90 degrees counter-clockwise]
        degrees = ["", "'"]
        degrees_inference = degrees[::-1]
        self.moves = [f"{f}{n}" for f in faces for n in degrees]
        self.moves_inference = [f"{f}{n}" for f in faces for n in degrees_inference]

        # Opposite faces
        self.pairing = {
            "R": "L",
            "L": "R",
            "F": "B",
            "B": "F",
            "U": "D",
            "D": "U",
        }
        # Prohibit obviously redundant moves.
        self.moves_available_after = {
            m: [v for v in self.moves if v[0] != m[0]] + [m] 
            for m in self.moves
        } # self-cancelling moves on the same face

        # [OPTIMIZATION] slicing by move string (e.g., R', U, F) => indices (e.g., 2, 6, 1)
        self.moves_ix = [self.moves.index(m) for m in self.moves]
        self.moves_ix_available_after = {
            self.moves.index(m): [self.moves.index(m) for m in available_moves]
            for m, available_moves in self.moves_available_after.items()
        }
        self.moves_ix_inference = [self.moves.index(m) for m in self.moves_inference]
        self.pairing_ix = {
            0: 1,
            1: 0,
            2: 3,
            3: 2,
            4: 5,
            5: 4,
        } # Points to the opposite face index

        # Vectorize the sticker group replacement operations
        self.__vectorize_moves()

    def reset(self):
        """Resets the cube state to the solved state."""
        self.state = np.arange(0, 9 * 6, dtype=self.DTYPE) // 9

    def is_solved(self):
        """Checks if the cube is in the solved state."""
        return np.all(self.state == self.goal)

    def finger(self, move):
        """Applies a single move on the cube state using move string."""
        self.state[self.sticker_target[move]] = self.state[self.sticker_source[move]]

    def finger_ix(self, ix):
        """The same `finger` method **but using indices of moves for faster execution"""
        self.state[self.sticker_target_ix[ix]] = self.state[self.sticker_source_ix[ix]]

    def apply_scramble(self, scramble):
        """Applies a sequence of moves (scramble) to the cube state."""
        if isinstance(scramble, str):
            scramble = scramble.split()
        for m in scramble:
            if m[-1]=='2':
                for _ in range(2):
                    self.finger(m[0])
            else:
                    self.finger(m)

    def scrambler(self, scramble_length):
        """
        Generates a random scramble of given length and returns the cube state and scramble moves as a generator.
        Please note that index-based implementations (faster) follow commented lexical logics.
        """
        while True:
            # Reset the cube state, scramble, and return cube state and scramble moves
            self.reset()
            scramble = []

            for i in range(scramble_length):
                if i:
                    last_move = scramble[-1]
                    if i > 1:   # [3rd~ moves]
                        while True:
                            # move = random.choice(self.moves_available_after[last_move])
                            move = random.choice(self.moves_ix_available_after[last_move])

                            if scramble[-2] == last_move == move:
                                # Three subsequent moves on the same face, which could be one
                                continue
                            # elif (
                            #     scramble[-2][0] == move[0] and len(scramble[-2] + move) == 3
                            #     and last_move[0] == self.pairing[move[0]]
                            # ):
                            elif (
                                scramble[-2]//2 == move//2 and scramble[-2]%2 != move%2
                                and last_move//2 == self.pairing_ix[move//2]
                            ):
                                # Two mutually canceling moves sandwiching an opposite face move
                                continue
                            else:
                                break
                    else:       # [2nd move]
                        # move = random.choice(self.moves_available_after[last_move])
                        move = random.choice(self.moves_ix_available_after[last_move])
                else:           # [1st move]
                    # move = random.choice(self.moves)
                    move = random.choice(self.moves_ix)

                # self.finger(move)
                self.finger_ix(move)
                scramble.append(move)

                yield self.state, move


    def __vectorize_moves(self):
        """
        Vectorizes the sticker group replacement operations for faster computation.
        This method defines ```self.sticker_target``` and ```self.sticker_source``` to manage sticker colors (target is replaced by source).
        They define indices of target and source stickers so that the moves can be vectorized.

        Colors:

                0 0 0
                0 0 0
                0 0 0
        2 2 2   5 5 5   3 3 3   4 4 4
        2 2 2   5 5 5   3 3 3   4 4 4
        2 2 2   5 5 5   3 3 3   4 4 4
                1 1 1
                1 1 1
                1 1 1

        Order of stickers on each face:

             2   5   8
             1   4   7
            [0]  3   6

        Indices of state (each starting with 9*(n-1)):

                         2   5   8
                         1   4   7
                        [0]  3   6
             20  23 26  47  50  53  29  32 35  38  41 44
             19  22 25  46  49  52  28  31 34  37  40 43
            [18] 21 24 [45] 48  51 [27] 30 33 [36] 39 42
                        11   14 17
                        10   13 16
                        [9]  12 15
        """
        self.sticker_target, self.sticker_source = dict(), dict()

        self.sticker_replacement = {
            # Sticker A is replaced by another sticker at index B -> A:B
            'U':{0: 6, 1: 3, 2: 0, 3: 7, 5: 1, 6: 8, 7: 5, 8: 2, 20: 47, 23: 50, 26: 53, 29: 38, 32: 41, 35: 44, 38: 20, 41: 23, 44: 26, 47: 29, 50: 32, 53: 35},
            'D':{9: 15, 10: 12, 11: 9, 12: 16, 14: 10, 15: 17, 16: 14, 17: 11, 18: 36, 21: 39, 24: 42, 27: 45, 30: 48, 33: 51, 36: 27, 39: 30, 42: 33, 45: 18, 48: 21, 51: 24},
            'L':{0: 44, 1: 43, 2: 42, 9: 45, 10: 46, 11: 47, 18: 24, 19: 21, 20: 18, 21: 25, 23: 19, 24: 26, 25: 23, 26: 20, 42: 11, 43: 10, 44: 9, 45: 0, 46: 1, 47: 2},
            'R':{6: 51, 7: 52, 8: 53, 15: 38, 16: 37, 17: 36, 27: 33, 28: 30, 29: 27, 30: 34, 32: 28, 33: 35, 34: 32, 35: 29, 36: 8, 37: 7, 38: 6, 51: 15, 52: 16, 53: 17},
            'B':{2: 35, 5: 34, 8: 33, 9: 20, 12: 19, 15: 18, 18: 2, 19: 5, 20: 8, 33: 9, 34: 12, 35: 15, 36: 42, 37: 39, 38: 36, 39: 43, 41: 37, 42: 44, 43: 41, 44: 38},
            'F':{0: 24, 3: 25, 6: 26, 11: 27, 14: 28, 17: 29, 24: 17, 25: 14, 26: 11, 27: 6, 28: 3, 29: 0, 45: 51, 46: 48, 47: 45, 48: 52, 50: 46, 51: 53, 52: 50, 53: 47}
        }
        for m in self.moves:
            if len(m) == 1:
                assert m in self.sticker_replacement
            else:
                if "'" in m:
                    self.sticker_replacement[m] = {
                        v: k for k, v in self.sticker_replacement[m[0]].items()
                    }
                elif "2" in m:
                    self.sticker_replacement[m] = {
                        k: self.sticker_replacement[m[0]][v]
                        for k, v in self.sticker_replacement[m[0]].items()
                    }
                else:
                    raise

            self.sticker_target[m] = list(self.sticker_replacement[m].keys())
            self.sticker_source[m] = list(self.sticker_replacement[m].values())

            for i, idx in enumerate(self.sticker_target[m]):
                assert self.sticker_replacement[m][idx] == self.sticker_source[m][i]

        # For index slicing
        self.sticker_target_ix = np.array([np.array(self.sticker_target[m]) for m in self.moves])
        self.sticker_source_ix = np.array([np.array(self.sticker_source[m]) for m in self.moves])


def load_environment():
    return Cube3()

if __name__=="__main__":
    env = Cube3()
    print(f"Goal:\n{env.state.reshape(6,9)}")
    env.apply_scramble("R R F U' B F D L D R'")
    print(f"Scrambled:\n{env.state.reshape(6,9)}")
