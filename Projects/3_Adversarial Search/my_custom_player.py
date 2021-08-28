
from collections import defaultdict
from io import UnsupportedOperation
from logging import currentframe
from os import POSIX_FADV_DONTNEED, stat
from pickle import READONLY_BUFFER
from typing import DefaultDict, List
from warnings import resetwarnings
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
        self.children_of = defaultdict(list) # child relationships children[state] = [state, state2]
        self.frontier = set()
        self.exploration_weight = 1
        self.parent_of = dict()
        self.root = None 

    def _reset(self, root):
        self.root = root 
        self.frontier.clear()
        self.frontier.add(root)        
        self.n[root] = 0
        self.q[root] = 0        
        self.parent_of.clear()
        self.parent_of[root] = None
        self.children_of.clear()

    def expand(self, state):
        """adding a random new, unvisited and unrated child to the tree"""
        assert(state in self.frontier)
        if state in self.frontier: self.frontier.remove(state)
        for idx, a in enumerate(state.actions()):
            child = state.result(a)
            self.children_of[state].append(child)
            self.parent_of[child] = state
            self.n[child] = 0
            self.q[child] = 0
            self.frontier.add(child)
        
    def rollout(self, state): #simulation
        front_state = copy.copy(state) ## copy maybe necessary?
        while True:            
            if front_state.terminal_test():
                return 1 if front_state.utility(self.player_id) > 0 else -1
            front_state = random.choice([front_state.result(a) for a in front_state.actions() if front_state.result(a) not in self.children_of.keys()])        

    def ucb(self, state):
        c = self.exploration_weight # to be tweaked?
        parent_n = self.n[self.parent_of[state]] if self.parent_of[state] else 0        
        if self.n[state] == 0:
            return float("-inf")  # avoid unvetted moves
        return (self.q[state] + c * math.sqrt(2 * math.log(2*parent_n / self.n[state])))
    

    def select_next_node(self):
        #Ranking children by policy        
        return max(self.frontier, key=self.ucb)  


    def update(self, reward, explored_state):
        state = copy.copy(explored_state)
        while state:            
            self.n[state] += 1
            self.q[state] += reward            
            state = self.parent_of[state]

    def evaluate_by_simulation(self, from_nodes: list, number_of_sims: int):
        for _ in range(number_of_sims):                                       
            nxt = random.choice(from_nodes)  # choose a child for rollout
            reward = self.rollout(nxt) # go to a terminal node and see if we win
            self.update(reward, nxt) #updata the tree from root to state_to_explore
            #print(f"{len(self.frontier)=}")

    def run(self, root):
        self._reset(root)
        state = root
        for round in range(3) :
            if  state.terminal_test():                
                reward = 1 if state.utility(self.player_id) > 0 else -1
                self.update(reward, state)
                break
            self.expand(state) # add all children            
            self.evaluate_by_simulation(self.children_of[state], 40)
            state = self.select_next_node() # etither the best or the least explored            

        return max(root.actions(), key= lambda a: self.q[root.result(a)])

                
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
        
        move = self.tree.run(state)
        self.queue.put(move)
