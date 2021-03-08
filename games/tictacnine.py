import copy
import datetime
import os
import random

import numpy
import torch

from .abstract_game import AbstractGame

def pc(ch):
    if ch == 1:
        return 'X'
    if ch == -1:
        return 'O'
    return '_'

configuration_4 = {
    'seed' : 0,
    'opponent': 'expert',
    'max_moves': 81,
    'num_simulations': 50,
    'temperature_threshold': 30,
    'channels': 128,
    'reduced_channels_reward': 16,
    'reduced_channels_value': 16,
    'reduced_channels_policy': 16,
    'resnet_fc_reward_layers': [64],
    'resnet_fc_value_layers': [64],
    'resnet_fc_policy_layers': [64],
    'encoding_size':32,
    'fc_representation_layers': [32],
    'fc_dynamics_layers': [64],
    'fc_reward_layers': [64],
    'fc_value_layers': [32],
    'fc_policy_layers': [32],
    'training_steps':10000,
    'batch_size':256,
    'checkpoint_interval':25,
    'optimizer': 'SGD',
    'replay_buffer_size': 6000,
    'num_unroll_steps':20,
    'td_steps':20,
    'lr_init' : 0.003,
    'lr_decay_rate': 1,
    'support_size':10,
}

configuration_5 = {
    'seed' : 1337,
    'opponent': 'expert',
    'max_moves': 81,
    'num_simulations': 100,
    'temperature_threshold': 30,
    'channels': 32,
    'reduced_channels_reward': 4,
    'reduced_channels_value': 4,
    'reduced_channels_policy': 8,
    'resnet_fc_reward_layers': [32],
    'resnet_fc_value_layers': [32],
    'resnet_fc_policy_layers': [32],
    'encoding_size':32,
    'fc_representation_layers': [16],
    'fc_dynamics_layers': [32],
    'fc_reward_layers': [32],
    'fc_value_layers': [16],
    'fc_policy_layers': [16],
    'training_steps':10000,
    'batch_size':256,
    'checkpoint_interval':25,
    'optimizer': 'SGD',
    'replay_buffer_size': 6000,
    'num_unroll_steps':20,
    'td_steps':20,
    'lr_init' : 0.002,
    'lr_decay_rate': 0.997,
    'support_size':4,
}

class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        active_configuration = configuration_5
        print(active_configuration);

        self.seed = active_configuration['seed']  # Seed for numpy, torch and the game
        self.max_num_gpus = 1 # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (3, 9,9)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(9 * 9))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = active_configuration['opponent']  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class
        # self.opponent = "random"

        ### Self-Play
        self.num_workers = 4  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = 1
        self.max_moves = active_configuration['max_moves'] # Maximum number of moves if game is not finished before
        self.num_simulations = active_configuration['num_simulations']# Number of future moves self-simulated
        self.discount = 1 #0.997  # Chronological discount of the reward
        self.temperature_threshold = active_configuration['temperature_threshold']# Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = active_configuration['support_size'] #10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 6  # Number of blocks in the ResNet
        self.channels = active_configuration['channels'] # Number of channels in the ResNet
        self.reduced_channels_reward = active_configuration['reduced_channels_reward']# Number of channels in reward head
        self.reduced_channels_value = active_configuration['reduced_channels_value']  # Number of channels in value head
        self.reduced_channels_policy = active_configuration['reduced_channels_policy']# Number of channels in policy head
        self.resnet_fc_reward_layers = active_configuration['resnet_fc_reward_layers']  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = active_configuration['resnet_fc_reward_layers']  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = active_configuration['resnet_fc_policy_layers']  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = active_configuration['encoding_size']
        self.fc_representation_layers = active_configuration['fc_representation_layers']  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = active_configuration['fc_dynamics_layers'] # Define the hidden layers in the dynamics network
        self.fc_reward_layers = active_configuration['fc_reward_layers']  # Define the hidden layers in the reward network
        self.fc_value_layers = active_configuration['fc_value_layers']# Define the hidden layers in the value network
        self.fc_policy_layers = active_configuration['fc_policy_layers']  # Define the hidden layers in the policy network

        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = active_configuration['training_steps']# Total number of training steps (ie weights update according to a batch)
        self.batch_size = active_configuration['batch_size']# Number of parts of games to train on at each training step
        self.checkpoint_interval = active_configuration['checkpoint_interval'] # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = active_configuration['optimizer']  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = active_configuration['lr_init']  # Initial learning rate
        self.lr_decay_rate = active_configuration['lr_decay_rate']  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.replay_buffer_size = active_configuration['replay_buffer_size']#3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = active_configuration['num_unroll_steps'] #20  # Number of game moves to keep for every batch element
        self.td_steps = active_configuration['td_steps'] #20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = True



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1.0
        # if trained_steps < 0.5 * self.training_steps:
            # return 1.0
        # elif trained_steps < 0.75 * self.training_steps:
            # return 0.5
        # else:
            # return 0.25

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = TicTacNine()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 20, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        while True:
            try:
                action_number = int(
                    input(
                        f"Enter the move number (0 to 80) to play for the player {self.to_play()}: "
                    )
                )
                choice = action_number
                if (
                    choice in self.legal_actions()
                    and 0 <= action_number
                    and action_number <= 80
                ):
                    break
            except:
                pass
            print("Wrong input, try again")
        return choice

    def action_to_string(self, action_number): # type: ignore
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return f"Play on field {action_number}"


class TicTacNine:
    def __init__(self):
        self.board = numpy.zeros((9,9), dtype="int32")
        self.player = 1
        self.last_move = None

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = numpy.zeros((9,9), dtype="int32")
        self.player = 1
        self.last_move = None
        return self.get_observation()

    def step(self, action):
        row = action // 9
        col = int(action % 9)
        self.board[row][col] = self.player
        self.last_move = action

        done = self.have_winner() or len(self.legal_actions()) == 0

        reward = 1 if self.have_winner() else 0

        # @TODO maybe improvement? grant a few points if game is draw?

        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1, 0)
        board_player2 = numpy.where(self.board == -1, 1, 0)
        board_to_play = numpy.full((9,9), self.player)
        return numpy.array([board_player1, board_player2, board_to_play], dtype="int32")

    def legal_actions(self):
        legal = []
        board = self.board.reshape(81)
        allowed_fields = playable_fields(board)

        for i in range(81):
            last_move = None
            if self.last_move != None:
                last_move = create_move(-1, self.last_move)

            if is_valid_next_move(board,last_move, create_move(1, i), allowed_fields):
                legal.append(i)

        return legal

    def expert_action(self):
        board = copy.deepcopy(self.board).reshape(81)
        legal_actions = self.legal_actions();
        best_action = legal_actions[0];
        expert = self.player *-1;

        score = -20

        for legal_action in self.legal_actions():
            board[legal_action] = expert;

            # if this moves wins the game for use, best move!
            if is_won(board) == expert:
                return legal_action

            # (1a) a bit of randomness for each move, just in case
            # they fulfill the same criteria
            move_score =random.randint(0,9)

            forced_field = int(legal_action % 9)
            target_field = board[forced_field:forced_field+9]

            # bias (Experimental) personally I have the feeling
            # that it is a good strategy to drop the middle and try
            # to win elsewhere, the question is now, is that a good
            # bot strategy? Let's test three biases, center, even odd fields, or corners

            # (1b) center bias
            if forced_field == 4:
                    move_score += 3

            # (1b) corner bias could also be combined with center bias
            # just drop the first condition then
            # if forced_field != 4 and forced_field %2 == True:
                # move_score += 3

            # (1b) even bias
            # if forced_field %2 == False:
                # move_score += 3

            # (2a) if target field results in player being able to chose
            # deduct points
            if is_full_field(target_field) or field_winner(target_field) != 0:
                move_score -= 25
            else:
                # (2b) if field is still playable, but the "stay here" field has already been played
                # award some points
                if target_field[forced_field] != 0:
                    move_score += 7

            played_field = int(legal_action / 9)

            # (3a) will this move let me win this field?
            if field_winner(board[played_field:played_field+9]) == expert:
                move_score += 8
            else:
                field = board[played_field*9:played_field*9+9]
                # (3b) will this move block a row? we assume that it is a good thing, because
                # it either is a mixed row which is nice to be blocked, or not, and then its nice
                # because we win the field
                for line in winLines:
                  if field[line[0]] != 0 and field[line[1]] != 0 and field[line[2]]!= 0:
                      if line[0] == forced_field or line[1] == forced_field or line[2] == forced_field:
                          move_score += 3

            # (4) a good strategy seems to be to play the move that forces the opponent to stay within this field
            if legal_action % 10 == 0:
                move_score += 10

            if move_score > score:
                score = move_score
                best_action = legal_action

            board[legal_action] = 0;

        return best_action


    def have_winner(self):
        return is_won(self.board.reshape(81)) == self.player


    def render(self):
        board = self.board.reshape(81)
        for i in range(3):
            m = i * 3
            print(f'{pc(board[m])}', end='')
            print(f'{pc(board[1+m])}', end='')
            print(f'{pc(board[2+m])}', end='')
            print("|", end='')

            print(f'{pc(board[9+m])}', end='')
            print(f'{pc(board[10+m])}', end='')
            print(f'{pc(board[11+m])}', end='')

            print("|", end='')

            print(f'{pc(board[18+m])}', end='')
            print(f'{pc(board[19+m])}', end='')
            print(f'{pc(board[20+m])}', end='')
            print("|\n", end='')

        print("----------------------\n", end='')

        for i in range(3):
            m = i * 3
            print(f'{pc(board[27+m])}', end='')
            print(f'{pc(board[28+m])}', end='')
            print(f'{pc(board[29+m])}', end='')
            print("|", end='')

            print(f'{pc(board[36+m])}', end='')
            print(f'{pc(board[37+m])}', end='')
            print(f'{pc(board[38+m])}', end='')
            print("|", end='')

            print(f'{pc(board[45+m])}', end='')
            print(f'{pc(board[46+m])}', end='')
            print(f'{pc(board[47+m])}', end='')
            print("|\n", end='')

        print("----------------------\n", end='')

        for i in range(3):
            m = i * 3
            print(f'{pc(board[54+m])}', end='')
            print(f'{pc(board[55+m])}', end='')
            print(f'{pc(board[56+m])}', end='')
            print("|", end='')

            print(f'{pc(board[63+m])}', end='')
            print(f'{pc(board[64+m])}', end='')
            print(f'{pc(board[65+m])}', end='')
            print("|", end='')

            print(f'{pc(board[72+m])}', end='')
            print(f'{pc(board[73+m])}', end='')
            print(f'{pc(board[74+m])}', end='')
            print("|\n", end='')

        print("----------------------\n", end='')

# just for readability, won't make sense to change as long as winlines is hardcoded
BOARD_SIZE = 81
FIELD_SIZE = 9

winLines =[
    [0, 4, 8],
    [0, 1, 2],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [2, 4, 6],
    [3, 4, 5],
    [6, 7, 8],
  ]

# field must be an array of length 9
# it then checks simply if there is one line that
# actually is won
# if two players have a line, the first one will be yield
def field_winner(field):
  if len(field) != FIELD_SIZE:
    raise Exception(f'Field must be of length {FIELD_SIZE}')

  for line in winLines:
      if field[line[0]] == 0:
        continue
      if field[line[0]] == field[line[1]] and field[line[1]] == field[line[2]] and field[line[0]] != 0:
        return field[line[0]]

  return 0

# all stones placed already?
# does not check if that field has been won already
def is_full_field(field):
  if len(field) != FIELD_SIZE:
    raise Exception(f'Field must be of length {FIELD_SIZE}')

  for p in field:
    if p == 0:
      return False

  return True

# returns all fields that are still playable
# does not take into account the last move
def playable_fields(board):
  if len(board) != BOARD_SIZE:
    raise Exception(f'Board must be exactly {BOARD_SIZE} fields')

  fields = []

  for i in range(0, BOARD_SIZE, FIELD_SIZE):
    area = board[i:i+FIELD_SIZE]
    if not is_full_field(area) and field_winner(area) == 0:
      fields.append(int(i / 9))

  return fields

def create_move(color, position):
  return {'color': color, 'position': position}

def check_move(move):
  if type(move) is not dict:
    return False

  return 'color' in move and 'position' in move

def is_valid_next_move(board, last_move, next_move, allowed_fields):
  # if len(board) != BOARD_SIZE:
    # raise Exception(f'Board must be exactly {BOARD_SIZE} fields')
  # if last_move != None and check_move(last_move) == False:
    # raise Exception(f'Last move param invalid')
  # if check_move(next_move) == False:
    # raise Exception(f'Next move param invalid')

  # already a stone
  if board[next_move['position']] != 0: # hacked here..
    return False

  played_field = int(next_move['position'] / 9)

  if last_move != None:
    # cannot play twice
    if last_move['color'] == next_move['color']:
      return False

    # guard against same position, but should not be possible if board
    # is saved correctly
    if last_move['position'] == next_move['position']:
      return False

    forced_field = int(last_move['position'] % 9)
  else:
    forced_field = -1

  if len(allowed_fields) == 0:
    return False

  for allowed in allowed_fields:
    if forced_field == -1:
      return True

    if allowed == forced_field:
      if played_field == forced_field:
        return True
      else:
        return False

  for allowed in allowed_fields:
    if allowed == played_field:
      return True

  return False

# returns the winner of the whole board
def is_won(board):
  if len(board) != BOARD_SIZE:
    raise Exception(f'Board must be exactly {BOARD_SIZE} fields')

  big_board = [None] * 9
  for i in range(0, BOARD_SIZE, FIELD_SIZE):
    area = board[i:i+FIELD_SIZE]
    winner = field_winner(area)
    big_board[int(i / 9)] = winner # type: ignore

  return field_winner(big_board)

def ended(board):
  if len(board) != BOARD_SIZE:
    raise Exception(f'Board must be exactly {BOARD_SIZE} fields')

  winner = is_won(board)
  if winner == 1:
    return {'won': True, 'by': 1}
  if winner == -1:
    return {'won': True, 'by': -1}

  # draw?
  if len(playable_fields(board)) == 0:
    return {'won': True, 'by': 0}

  return {'won': False, 'by': 0}
