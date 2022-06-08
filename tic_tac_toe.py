import numpy as np

class Agent:
    def __init__(self):
        self.epsilon = 0.1  # probability based randon action is chosen
        self.alpha = 0.5  # learning rate
        self.hist_state = []  # history of all states

    def initialize_V(self, env, winning_state_triples):
        V = np.zeros(env.maximum_states)
        for state, winner, ended in winning_state_triples:
            if ended:
                if winner == env.x:  # Let's say agent is 'X'
                    state_value = 1 # if agent wins the game
                else:
                    state_value = 0 # if agent loose the game or game is a draw
            else:
                state_value = 0.5

            V[state] = state_value
        self.V = V

    def set_symbol(self, symbol):
        self.symbol = symbol

    def hist_reset(self): # to reset history
        self.hist_state = []

    def selecting_choose_random(self, env):
        vacant_moves = env.fetch_vacant_moves() # possible moves can be 0, 1 and 2
        vacant_moves_random_indexing = np.random.choice(len(vacant_moves))
        random_next_move = vacant_moves[vacant_moves_random_indexing]
        return random_next_move

    def favorable_outcome_from_states(self, env):
        next_favorable_outcome, favorable_state = env.get_next_favorable_outcome(self)
        return next_favorable_outcome, favorable_state

    def fetch_next_move(self, env):
        next_favorable_outcome, favorable_state = None, None # take a random decision or decision from history
        random_number = np.random.rand()  # float between 0 and 1
        if random_number < self.epsilon:
            # take a random action
            next_favorable_outcome = self.selecting_choose_random(env)
        else:
            next_favorable_outcome, favorable_state = self.favorable_outcome_from_states(env) # best action is chosen from current states
        return next_favorable_outcome, favorable_state

    def make_decision(self, env):
        selected_next_move, favorable_state = self.fetch_next_move(env)
        # make next move
        env.tic_tac_toe_panel[selected_next_move[0], selected_next_move[1]] = self.symbol

    def update_hist_state(self, state): # add each state to history to use in future
        self.hist_state.append(state)

    def update(self, env):
        # V(prev_state) = V(prev_state) + alpha * ( V(next_state) - V(pre_state) )
        # here, V(next_state) is prize
        prize = env.prize(self.symbol) # price is a word used for reward concept in reinforcement learning
        goal = prize
        for prev in reversed(self.hist_state):
            value = self.V[prev] + self.alpha * (goal - self.V[prev])
            self.V[prev] = value
            goal = value
        self.hist_reset()

class Environment:

    def __init__(self):
        self.tic_tac_toe_panel = np.zeros((3, 3))  # 2D array with zeroes
        self.x = -1  # player 1
        self.o = 1  # player 2
        self.winner = None  # no winner at beginning
        self.ended = False  # game can't be finished before its start
        self.maximum_states = 3 ** (3 * 3)  # calculating number of states possible in game (19683 states)

    def is_vacant(self, i, j):
        # it shows position (x,y) on board is vacant or not
        return self.tic_tac_toe_panel[i, j] == 0

    def prize(self, symbol):
        # agent is rewarded at the end of game
        collected_prize = 0
        if self.game_finish() and self.winner == symbol:  # winning player is rewarded as 1
            collected_prize = 1
        return collected_prize

    def is_tie(self):
        is_tie = False
        if self.ended and self.winner is None:  # draw scene
            is_tie = True
        return is_tie

    def fetch_state(self):
        # current state in integer is return value
        state = 0
        loop_index = 0
        for i in range(3):
            for j in range(3):
                if self.tic_tac_toe_panel[i, j] == self.x:
                    state_value = 1
                elif self.tic_tac_toe_panel[i, j] == self.o:
                    state_value = 2
                else:
                    state_value = 0  # empty

                state += (3 ** loop_index) * state_value
                loop_index += 1
        return state

    def game_finish(self):
        # returns 1 if player wins or there is a draw in the game
        if self.ended:  # return 1 if game ends
            return True

        players = [self.x, self.o]

        # verify for same symbols on row side
        for i in range(3):
            for player in players:
                if self.tic_tac_toe_panel[i].sum() == player * 3:  # results will be  1+1+1 = 3 for player 'O' and -1-1-1 = -3 for player 'X'
                    self.winner = player
                    self.ended = True
                    return True  # game finish

        # verify for same symbols on column side
        for j in range(3):
            for player in players:
                if self.tic_tac_toe_panel[:, j].sum() == player * 3:
                    self.winner = player
                    self.ended = True
                    return True  # game finish

        # verify for same symbols on diagonal side
        for player in players:
            # top-left to bottom-right diagonal
            if self.tic_tac_toe_panel.trace() == player * 3:
                self.winner = player
                self.ended = True
                return True  # game finish

            # top-right to bottom-left diagonal
            if np.fliplr(self.tic_tac_toe_panel).trace() == player * 3:
                self.winner = player
                self.ended = True
                return True  # game finish

        tic_tac_toe_panel_with_true_false = self.tic_tac_toe_panel == 0
        if np.all(tic_tac_toe_panel_with_true_false == False):
            # draw scene, so there is no winner
            self.winner = None
            self.ended = True
            return True  # # game finish

        # if game is incomplete
        self.winner = None
        return False

    def fetch_vacant_moves(self):
        vacant_moves = []
        # tracing all 9 boxes and collect empty boxes
        for i in range(3):
            for j in range(3):
                if self.is_vacant(i, j):  # check for empty box
                    vacant_moves.append((i, j))
        return vacant_moves

    def get_next_favorable_outcome(self, agent):
        # symbol can be 'X' or 'O'
        # select the best empty box
        best_value = -1
        next_favorable_outcome = None
        favorable_state = None
        for i in range(3):
            for j in range(3):
                if self.is_vacant(i, j):
                    # check the state if we select (x,y) position in game box
                    self.tic_tac_toe_panel[i, j] = agent.symbol
                    state = self.fetch_state() # temporary step and check state
                    self.tic_tac_toe_panel[i, j] = 0  # revert back to empty state ie actual state
                    if agent.V[state] > best_value:
                        best_value = agent.V[state]
                        favorable_state = state
                        next_favorable_outcome = (i, j)

        return next_favorable_outcome, favorable_state

    def draw_tic_tac_toe_panel(self): # tic-tac-toe display board
        def __print(to_print, j):
            if j == 0:
                print(f"|  {to_print}  ", end="|")
            else:
                print(f"{to_print}  ", end="|")

        for i in range(3):
            print(" ---------------------")
            for j in range(3):
                print("  ", end="")
                if self.tic_tac_toe_panel[i, j] == self.x:
                    __print('x', j)
                elif self.tic_tac_toe_panel[i, j] == self.o:
                    __print('o', j)
                else:
                    __print(' ', j)
            print("")
        print(" ---------------------")
        print("\n")


class Human:

    def set_symbol(self, symbol):
        self.symbol = symbol

    def make_decision(self, env):
        # till the legal move played by human
        while True:
            try:
                move = input("Enter box location for your move in (i,j) format : ")
                i, j = [int(item.strip()) for item in move.split(',')]
                if env.is_vacant(i, j):
                    env.tic_tac_toe_panel[i, j] = self.symbol
                    break
                else:
                    print("Please enter valid move")
            except:
                print("Please enter valid move")


def fetch_state_hash_and_winner(env, i=0, j=0):
    # recursive function returning all possible states with the winners in these states if any
    results = []
    for v in [0, env.x, env.o]:
        env.tic_tac_toe_panel[i, j] = v  # if empty board, value is 0
        if j == 2:
            if i == 2:
                # full board with result
                state = env.fetch_state()
                ended = env.game_finish()
                winner = env.winner
                results.append((state, winner, ended))
            else:
                results += fetch_state_hash_and_winner(env, i + 1, 0)
        else:
            # j is inc, i is constant
            results += fetch_state_hash_and_winner(env, i, j + 1)
    return results


def play_tic_tac_toe(agent, human, env, print_tic_tac_toe_panel=True):
    current_player = None  # player1 starts game
    # till game is completed
    game_playing_continue = True


    while game_playing_continue:
        if current_player == agent:
            current_player = human
        else:
            current_player = agent

        # current player playing the move
        current_player.make_decision(env)

        # state histories updated
        if current_player == agent:
            state = env.fetch_state()
            agent.update_hist_state(state)  # player1 is agent
            agent.update(env)
            if print_tic_tac_toe_panel:
                env.draw_tic_tac_toe_panel()  # tic_tac_toe_panel is updated for draw scene

        if env.game_finish():
            game_playing_continue = False


def main(must_learn_prior_to_play):
    print("LET'S PLAY TIC TAC TOE GAME")
    print("Agent -> X")
    print("Human -> 0")

    # environment invoked
    env = Environment()

    winning_state_triples = fetch_state_hash_and_winner(env)

    # agent is player1
    agent = Agent()
    agent.set_symbol(env.x)
    agent.initialize_V(env, winning_state_triples)

    if must_learn_prior_to_play:
        print("Agent is learning by playing with itself...")
        # to learn
        agent_learning = Agent()
        agent_learning.set_symbol(env.o)
        agent_learning.initialize_V(env, winning_state_triples)

        for i in range(10000):
            if i > 0 and i % 1000 == 0:
                print(f"Agent has played {i} times")
            play_tic_tac_toe(agent, agent_learning, Environment(), print_tic_tac_toe_panel=False)
        print("")
        print("Agent has played 10,000 times to learn...")

    # game between agent and human
    human = Human()
    human.set_symbol(env.o)
    total_game_played = 0
    while True:
        env = Environment()
        play_tic_tac_toe(agent, human, env=env)

        total_game_played += 1
        print(f"Game number: {total_game_played}")
        if env.winner == env.x:
            print(f"Agent won the game")
        elif env.winner == env.o:
            print(f"You won the game")
        else:
            print(f"Game is draw")

        reply = input("Do you wish to play more? [y/n]: ")
        if reply and reply.lower()[0] == 'n':
            break

if __name__ == '__main__':
    main(must_learn_prior_to_play=True)