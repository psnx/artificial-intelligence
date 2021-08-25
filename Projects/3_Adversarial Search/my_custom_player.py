
from collections import defaultdict
from os import POSIX_FADV_DONTNEED, stat
from pickle import READONLY_BUFFER
from typing import DefaultDict, List
from isolation.isolation import S, Isolation
from sample_players import DataPlayer

#TODO: Remove this import for submission
from isolation import DebugState
import random
import math
import copy

class MCTS():
    def __init__(self, player_id) -> None:
        self.player_id = player_id
        self.q = defaultdict(int)
        self.n = defaultdict(int)
        self.children =  dict() # child relationships children[state] = [state, state2]
        self.exploration_weight = 1
        self.parent = dict() # dict when not root
  

    def find_an_unvisited_child(self, state):
        return random.choice([state.result(a) for a in state.actions() if a not in self.children[state] ])
        

    def expand(self, state):
        """adding a random new, unvisited and unrated child to the tree"""
        new_state = self.find_an_unvisited_child(state)
        self.children[state].append(new_state)
        self.n[new_state] = 0
        self.q[new_state] = 0
        self.parent[new_state] = state
        return new_state

    def select(self, state):
        pass

    def rollout(self, state): #simulation
        front_state = state ## copy maybe necessary?
        path = []
        while True:
            path.append(front_state)            
            if front_state.terminal_test():
                return path
            front_state = self.find_an_unvisited_child(front_state)

    def ucb(self, state):
        c = self.exploration_weight # to be tweaked?
        #if not self.parent : return float("inf")
        if self.n[state] == 0:
            return float("-inf")
        parent_n = self.n[self.parent[state]]        
        return (self.q[state] + c * math.sqrt(2 * math.log(2*parent_n / self.n[state])))
    

    def best_child(self, state):
        #Ranking children by policy 
        print(f"from best child {self.n[state]=}")
        return max(self.children[state], key=self.ucb)  


    def backpropagate(self, path, state):
        final = path.pop()
        self.q[final] = final.utility(state.player_id)
        self.n[final] += 1
        outcome = self.q[final]
        while path:
            up = path.pop()            
            self.n[up] += 1
            self.q[up] = outcome

    def reset(self):
        self.player_id = player_id
        self.q = defaultdict(int)
        self.n = defaultdict(int)
        self.children =  dict() # child relationships children[state] = [state, state2]
        self.exploration_weight = 1


    def run(self, state):
        self.children[state] = []
        for _ in range(100):
            if self.children[state]:
                nxt = self.best_child(state)
                self.expand(nxt)
            self.expand(state)
        result =  max([a for a in self.actions()], key=lambda x: self.q[x.result()] )
        self.reset()
        print(f"{self.q[state]=}")
        return result
        
                
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
      
    def __init__(self, player_id):
        super().__init__(player_id)
        self.tree = MCTS(self.player_id)

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
        # remove state
        move = self.tree.run(state)
        self.queue.put(move)
