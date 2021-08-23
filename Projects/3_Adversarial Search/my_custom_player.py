
from collections import defaultdict
from os import POSIX_FADV_DONTNEED
from pickle import READONLY_BUFFER
from typing import DefaultDict
from isolation.isolation import S, Isolation
from sample_players import DataPlayer

#TODO: Remove this import for submission
from isolation import DebugState
import random
import math

class Node:
    def __init__(self, state) -> None:
        self.children = set()
        self.parent = None
        self.visit_count = 0

    def is_terminal(self, state):
        return state.terminal_test()

    def reward(self, state):
        return state.utility()

class MCTS():
    def __init__(self, player_id) -> None:
        self.player_id = player_id
        self.q = defaultdict(int)
        self.n = defaultdict(int)
        self.children = dict() # child relationships
        self.exploration_weight = 1
        self.parent = None # dict when not root
        self.rewards = dict()

    def unvisited_children(self, state):
        self.children[state] = [(state.result(action)) for action in state.actions()]
        return [a for a in self.children[state] if self.n[a] == 0]
    
    def ucb(self, state):
        c = self.exploration_weight # to be tweaked?
        if not self.parent : return float("inf")
        return (self.q[state] + c * math.sqrt(2 * math.log(self.parent[state]) / self.n[state]))

    # this may be is not efficient...
    def find_children(self, state):
        if state.termnal_test():
            return None # we are at a leaf
        children = []
        for a in state.actions():
            result = state.result(a)
            self.parent[result] = state 
            children.append(result)                
        self.children[state] = children
        return children

    def random_child(self, state):
        random.choice([state.result(a) for a in state.actions()])

    def reward(self, state):
        if state.terminal_test():
            return state.utility(state.player_is)
        print("we should not have been here")
        return 0

    def best_child(self, state):
        #Ranking children by policy 
        return max(self.children[state], key=self.ucb)  

    def expand(self, state):
        new_state = random.choice(self.unvisited_children(state))
        self.children[new_state] = new_state
        return new_state

        
    def move(self, state):
        for c in state.actions():
            if state.result(c).terminal_test():
                if state.result(c).utility(self.player_id) == float("inf"):
                    return c

    def advance(self, state):
        while not state.terminal_test():
            if self.unvisited_children(state):
                return self.expand(state)
            state = self.best_child(state)
            return state

    def backpropagate(self, state):   
        r = state.utility(self.player_id)
        while state is not None:
            self.n[state] += 1
            #self.q[state] = ((node.n - 1)/node.n) * node.q + 1/node.n * r
            self.q[state] = ((self.n[state]-1) / self.n[state]) + 1/self.n[state] * r
            print(f"backpop: {state=} ")
            state = self.parent[state] if self.parent else None

    def main(self, state):
        for _ in range(100):
            next_state = self.advance(state)
            self.rewards[next_state] = self.ucb(next_state)
            self.backpropagate(state)
        #node should be root now... TODO: check
        return max([n for n in self.children[next_state]], key=lambda n: self.q[n])
                
class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
      

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        import random

        depth_limit = 1    
        #for depth in range(1, depth_limit+1):
        #    best_move = self.alpha_beta(state, depth)
        
        #print('In get_action(), state received:')
        #debug_board = DebugState.from_state(state)
        #print(debug_board)
        #print(state.ply_count)
        #move = (max(state.actions(), key=lambda n: self.score(n)))
        tree = MCTS(self.player_id)
        move = tree.main(state)
        print(f"next move {move=}")
        self.queue.put(move)
