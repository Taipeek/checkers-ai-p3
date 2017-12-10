import gamePlayReal as gamePlay
import numpy as np
from copy import deepcopy
from getAllPossibleMoves import getAllPossibleMoves
import sys

'''
The code makes use of recursion to implement minimax with alpha beta pruning.
'''


def evaluation(board, color, depth, turn, opponentColor, alpha, beta):
    if depth > 1:  # Comes here depth-1 times and goes to else for leaf nodes.
        depth -= 1
        opti = -sys.maxsize
        if turn == 'max':
            moves = getAllPossibleMoves(board, color)  # Gets all possible moves for player
            for move in moves:
                nextBoard = deepcopy(board)
                gamePlay.doMove(nextBoard, move)
                if beta > opti:
                    value = evaluation(nextBoard, color, depth, 'min', opponentColor, alpha, beta)
                    if value > opti:  # None is less than everything and anything so we don't need opti == None check
                        opti = value
                    if opti > alpha:
                        alpha = opti

        elif turn == 'min':
            moves = getAllPossibleMoves(board, opponentColor)  # Gets all possible moves for the opponent
            for move in moves:
                nextBoard = deepcopy(board)
                gamePlay.doMove(nextBoard, move)
                if alpha == None or opti == None or alpha < opti:  # None conditions are to check for the first times
                    value = evaluation(nextBoard, color, depth, 'max', opponentColor, alpha, beta)
                    if opti == None or value < opti:  # opti = None for the first time
                        opti = value
                    if opti < beta:
                        beta = opti

        return opti  # opti will contain the best value for player in MAX turn and worst value for player in MIN turn

    else:  # Comes here for the last level i.e leaf nodes
        state = np.zeros((1, 8, 8, 4))
        for i in range(0, 8):
            for j in range(0, 8):
                if board[i][j] == 'r':
                    state[0, i, j, 0] = 1
                elif board[i][j] == 'R':
                    state[0, i, j, 2] = 1
                elif board[i][j] == 'w':
                    state[0, i, j, 1] = 1
                elif board[i][j] == 'W':
                    state[0, i, j, 3] = 1
        value = gamePlay.model.predict(state)[:, 0][0]
        # print(value)
        return value


def nextMove(board, color, time, movesRemaining, prob):
    moves = getAllPossibleMoves(board, color)
    opponentColor = gamePlay.getOpponentColor(color)
    depth = 5
    best = None
    alpha = -sys.maxsize
    beta = float("inf")
    if np.random.random() > prob:
        return moves[np.random.randint(len(moves))]
    for move in moves:  # this is the max turn(1st level of minimax), so next should be min's turn
        newBoard = deepcopy(board)
        gamePlay.doMove(newBoard, move)
        # Beta is always inf here as there is no parent MIN node. So no need to check if we can prune or not.
        moveVal = evaluation(newBoard, color, depth, 'min', opponentColor, alpha, beta)
        if best == None or moveVal > best:
            bestMove = move
            best = moveVal
        if best > alpha:
            alpha = best
    return bestMove
