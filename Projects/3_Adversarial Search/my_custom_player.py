
from isolation.isolation import S, Isolation
from sample_players import DataPlayer

#TODO: Remove this import for submission
from isolation import DebugState

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
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        #own_xy = DebugState.ind2xy(own_loc)
        
        #(num_of_my_moves - 2 * num_of_opponent_moves) + (num_of_blank_spaces / 2 + 2 * num_of_my_moves) / 13
   

        return len(own_liberties) - len(opp_liberties)

    def max_value(self, state, depth, alpha, beta):
        """ Return the game state utility if the game is over,
        otherwise return the maximum value over all legal successors
        """        
        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <=0 : return self.score(state)
        v = float("-inf")
        for a in state.actions():
            v = max(v, self.min_value(state.result(a), depth - 1, alpha, beta))
            if v >= beta : return v
            alpha = max(alpha, v)
        return v

    def min_value(self, state, depth, alpha, beta):
        """ Return the game state utility if the game is over,
        otherwise return the minimum value over all legal successors
        """
        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <=0 : return self.score(state)
        v = float("inf")
        for a in state.actions():
            v = min(v, self.max_value(state.result(a), depth - 1, alpha, beta))
            if v <= alpha : return v
            beta = min(beta, v)
        return v

    #def minimax(self, state, depth):
    #    return max(state.actions(), key=lambda s: self.min_value(state.result(s), depth-1))

    def alpha_beta(self, state, depth):        
        alpha = float("-inf")
        beta = float("inf")  
        best_score = float("-inf")
        best_move = None      

        for a in state.actions():
            v = self.min_value(state.result(a), depth, alpha, beta)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a 
        return best_move        

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
        self.queue.put(self.alpha_beta(state, 2))
