# Checkers-game
An set of checkers playing AI agents, one of them with alpha-beta pruning.

The game execution starts by calling using the below command from console:

python gamePlay.py {-v} {-t time} player1 player2

-v is to enable verbose.
-t time is to specify maximum time available for each player.

player1 and player2 can be any of the below:

randomPlay

simpleGreedy

humanPlay

vpalakur

vpalakur-1


randomPlay is an AI agent that will randomly make a move.

simpleGreedy is an AI agent that will use a simple greedy approach and try to win the game.

humanPlay enables you to choose moves.

vpalakur is an AI agent which uses minimax with alpha-beta pruning to choose moves.

vpalakur-1 is built on top of vpalakur and changes its heuristic dynamically based on the remaining time.
