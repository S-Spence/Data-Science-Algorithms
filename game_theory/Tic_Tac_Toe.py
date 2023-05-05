# -*- coding: utf-8 -*-
import math
import random

"""
checkWin and whoWin functions
Created on Thu Sep 26 08:45:20 2019
@author: Samuel Fisher, Intern
Johns Hopkins University Applied Physics Laboratory
"""
#Display who won and add to win counter
def whoWin(x,End,Xwin,Owin): 
    if x == 1:
        End.configure(text="Player 1 has won!", background = 'white')
        Xwin = 1
    elif x == 2:
        End.configure(text="Player 2 has won!", background = 'white')
        Owin = 1
    else:
        End.configure(text="Nobody Wins", background = 'white')
    gameover = 1
    L = [Xwin,Owin,gameover]
    return L

#Check if there is a three in a row
#If there is a win, a display which team won and count that win
def checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill): 
    if place[1] == place[0] and place[0] == place[2] and place[1] != 0:
        print ("Player",place[1]," wins")
        return whoWin(place[1],End,Xwin,Owin)
    if place[0] == place[3] and place[0] == place[6] and place[0] != 0:
        print ("Player",place[0]," wins")
        return whoWin(place[0],End,Xwin,Owin)
    if place[0] == place[4] and place[0] == place[8] and place[0] != 0:
        print ("Player",place[0]," wins")
        return whoWin(place[0],End,Xwin,Owin)
    if place[1] == place[4] and place[1] == place[7] and place[1] != 0:
        print ("Player",place[1]," wins")
        return whoWin(place[1],End,Xwin,Owin)
    if place[2] == place[4] and place[2] == place[6] and place[2] != 0:
        print ("Player",place[2]," wins")
        return whoWin(place[2],End,Xwin,Owin)
    if place[2] == place[5] and place[2] == place[8] and place[2] != 0:
        print ("Player",place[2]," wins")
        return whoWin(place[2],End,Xwin,Owin)
    if place[3] == place[4] and place[3] == place[5] and place[3] != 0:
        print ("Player",place[3]," wins")
        return whoWin(place[3],End,Xwin,Owin)
    if place[6] == place[7] and place[8] == place[6] and place[6] != 0:
        print ("Player",place[6]," wins")
        return whoWin(place[7],End,Xwin,Owin)
    tie = 1
    for i in place:
        if i == 0:
            tie = 0
    if tie == 1:
        return whoWin(3,End,Xwin,Owin)
        
    return [0,0,0]

"""
Code below for PA1 problem 3
Author: Sarah Spence
Johns Hopkins University

"""
def checkWinPos(state):
    """Determine the next move for the AI"""
    eval_state = [-math.inf for i in range(9)]
    """
    Determine the possible moves from each point given that the board 
    is labeled as the example below.
    0, 3, 6
    1, 4, 7
    2, 5, 8
    """
    moves = {
            0: [[1, 2], [3, 6], [4, 8]],
            1: [[0, 2], [4, 7]],
            2: [[1, 0], [5, 8], [4, 6]],
            3: [[0, 6], [4, 5]],
            4: [[0, 8], [2, 6], [1, 7], [3, 5]],
            5: [[3, 4], [2, 8]],
            6: [[0, 3], [2, 4], [7, 8]],
            7: [[1, 4], [6, 8]],
            8: [[0, 4], [2, 5], [6, 7]]
            }
    
    for indx in range(len(state)):
        if state[indx] == 0:
            [value, can_win] = eval_function(indx, state, moves)
            
            if can_win == True:
                return indx
            eval_state[indx] = value
    
    best_move = max(eval_state)
    best_move_indices = []
    
    # Determine if there are multiple best moves
    for indx, val in enumerate(eval_state):
        if val == best_move:
            best_move_indices.append(indx)
    
    # Reevaluate to determine the best move if there are multiple best moves
    if len(best_move_indices) > 1:
        best_move_index = reeval_function(best_move_indices, state, moves)
    else:
        best_move_index = best_move_indices[0]
    return best_move_index
        
def reeval_function(best_move_indices: list, state: list, moves: dict) -> int:
    """If there are two best moves, choose a move to best block the other player. Randomize responses to be less predictable."""
    for indx in best_move_indices:
        # check if the center is open, if so, block it to prevent diagonal wins from opponent
        if indx == 4:
            return indx
        # If the center is not open, find a move that can block the other player
        for val in moves[indx]:
            # If the other player has an x in this row. This will never be called if there is a row with 2xs because the block will occur
            if val == 1:
                return indx
    # If no move can block the other player, return any move randomized
    random_move = random.choice(best_move_indices)
    return random_move
                  
def eval_function(point: int, state: list, moves: dict) -> [int, bool]:
    """
    Determine the ranking for a single point on the board using an eval function.
    If the other player can win, return true for a blocking move because this move is required.
    """
    # Values to track the number of xs and os in all possible directions from a single point
    x2 = 0
    o2 = 0
    x1 = 0
    o1 = 0
    
    # Boolean to specify if this move is a block or a win. Add 20 points for a block so this gets higher priority.
    block = False
    win = False
    
    # Look in all directions from this point and counts xs and os
    for direction in moves[point]:
        x = 0
        o = 0
        
        # determine which value to increment
        if state[direction[0]] == 1:
            x += 1
        if state[direction[0]] == 2:
            o += 1
        if state[direction[1]] == 1:
            x += 1
        if state[direction[1]] == 2:
            o += 1
        
        # increment the appropriate values for this direction
        if o == 2:
            o2 += 1
            win = True
        if o == 1:
            o1 += 1
        if x == 2:
            x2 += 1
            block = True
        if x == 1:
            x1 += 1
    
    # run the eval function to determine the ranking value of this point
    point_value = 3 * x2 + x1 - (3 * o2 + o1)
    # Add 20 points to total if this move is a block so it gets higher priority than other moves
    if block == True:
        point_value += 20
    
    return [point_value, win]   

"""
Code below for PA2 problem 3
Author: Sarah Spence
Johns Hopkins University
""" 
def decision_maker(boardState,minimax,depth):
    moves = {
        0: [[1, 2], [3, 6], [4, 8]],
        1: [[0, 2], [4, 7]],
        2: [[1, 0], [5, 8], [4, 6]],
        3: [[0, 6], [4, 5]],
        4: [[0, 8], [2, 6], [1, 7], [3, 5]],
        5: [[3, 4], [2, 8]],
        6: [[0, 3], [2, 4], [7, 8]],
        7: [[1, 4], [6, 8]],
        8: [[0, 4], [2, 5], [6, 7]]
        }
    
    # determine if a block is required
    actions = get_actions(boardState)
    blocking_move = must_block(boardState, actions, moves)
    
    if blocking_move != None:
        return blocking_move
    # If a block is not required, run min/max
    val, move = max_value(boardState, moves)
    return move

def max_value(state, moves):
    # check if game is terminal
    check = is_terminal(state, moves)
    # return the correct value if the game is terminal
    if check != False:
        return check, None
    
    
    actions = get_actions(state)
    max_v = math.inf * (-1)
    move = None
    
    for action in actions:
        # Player o takes moves in max
        state[action] = 2
        v2, a2 = min_value(state, moves)
        # Reset board state
        state[action] = 0
        if v2 > max_v:
            max_v, move =  v2, action
    return max_v, move
         
def min_value(state, moves):
    # check if game is terminal
    check = is_terminal(state, moves)
    # return the correct value if the game is terminal
    if check != False:
        return check, None
    
    actions = get_actions(state)
    min_v = math.inf
    move = None
   
    for action in actions:
        # Player x takes moves in min
        state[action] = 1
        v2, a2 = max_value(state, moves)
        state[action] = 0
        if v2 < min_v:
            min_v, move = v2, action
               
    return min_v, move

def must_block(state, actions, moves):
    """Determine if the algorithm needs to block before running min/max"""
    for action in actions:
        for move in moves[action]:
            if state[move[0]] == 1 and state[move[1]] == 1:
                return action
    return None

def is_terminal(state, moves):
    """Determine if the move is terminal"""
    for position in moves.keys():
        for win in moves[position]:
            # If x wins, return -1
            if state[position] == 1 and state[win[0]] == 1 and state[win[1]] == 1:
                return  -1
            # If o wins, return 1
            elif state[position] == 2 and state[win[0]] == 2 and state[win[1]] == 2:
                return 1
    
    # check for draws and return 0 for a draw
    open_pos = 9
    for val in state:
        if val != 0:
            open_pos -= 1

    if open_pos == 0:
        return 0

    return False
                     
def get_actions(state):
    actions = []
    for i in range(len(state)):
        if state[i] == 0:
            actions.append(i)
    return actions