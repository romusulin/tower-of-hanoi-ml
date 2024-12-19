class TowerOfHanoi:
    def __init__(self, num_disks):
        self.totalMoves = 0
        self.num_disks = num_disks
        self.pegs = [list(range(num_disks, 0, -1)), [], []]
        self.goal_state = [[], [], list(range(num_disks, 0, -1))]

    def reset(self):
        self.pegs = [list(range(self.num_disks, 0, -1)), [], []]
        self.totalMoves = 0
        return self.get_state()

    def get_state(self):
        return tuple(tuple(peg) for peg in self.pegs)

    def is_valid_move(self, from_peg, to_peg):
        if not self.pegs[from_peg]:
            return False
        if not self.pegs[to_peg] or self.pegs[from_peg][-1] < self.pegs[to_peg][-1]:
            return True
        return False

    def move_disk(self, from_peg, to_peg):
        self.totalMoves += 1
        if self.is_valid_move(from_peg, to_peg):
            disk = self.pegs[from_peg].pop()
            self.pegs[to_peg].append(disk)
            return True
        return False

    def get_possible_actions(self):
        actions = []
        for from_peg in range(3):
            for to_peg in range(3):
                if from_peg != to_peg:
                    actions.append((from_peg, to_peg))
        return actions

    def get_reward(self):
        # Reward based on how close the configuration is to the goal state
        if self.pegs == self.goal_state:
            return 10000  # Large reward if solved
        return -100  # Small negative reward for each move to encourage fewer steps

    def is_done(self):
        # Check if the puzzle is solved
        return self.pegs == self.goal_state

    def get_counter(self):
        return self.totalMoves
