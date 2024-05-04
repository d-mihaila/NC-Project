from random import random
import random as random_module
import numpy as np
import pygame

random_module.seed(42)

# ----------------simulation-----------------
numRows, numCols = 10, 10
snake = [(3, 3)]
apple = (5, 5)
v_x, v_y = 1, 0 
dead = False
NUM_ITERS = 10
# MAX_TIME_STEPS = 1000
MIN_TIME_TO_EAT_APPLE = 100

# ----------------animation stuff--------------
interval = 100
NODE_SIZE = 25
networkWidth, networkHeight = 700, 900
gameWidth, gameHeight = 900, 900
window_buffer = 25
screenWidth = window_buffer + networkWidth + window_buffer + gameWidth + window_buffer
screenHeight = networkHeight + 2 * window_buffer
blockWidth, blockHeight = gameWidth / numCols, gameHeight / numRows
gameTopLeft = (2 * window_buffer + networkWidth, window_buffer)
screen = None
RED = (255, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
BLUE = (70, 130, 180)
ORANGE = (255, 165, 13)
BUFFER = 8
font = None
# ----------------animation stuff--------------

def reset():
  global numRows
  global numCols
  global snake
  global apple
  global v_x
  global v_y
  global dead
  snake = [((int) (random() * numCols), (int) (random() * numRows))]
  apple = (int) (random() * numCols), (int) (random() * numRows)
  v_x, v_y = 1, 0 
  dead = False

def simulate_headless(net):
    ''' Simulate the game without GUI to train or evaluate the neural network. '''

    scores = []  # List to keep track of scores (snake length) for each iteration

    for _ in range(NUM_ITERS):  # Loop over a number of game iterations to evaluate performance
        reset()  # Reset the game environment to start state
        last_ate_apple = 0  # Track the last time step when an apple was eaten
        t = 0  # Initialize time step counter

        while True:  # Continue until snake dies or too much time passes without eating an apple
            if dead:  # Check if the snake has died (hit itself or a wall)
                break
            if t - last_ate_apple > MIN_TIME_TO_EAT_APPLE:  # Break if snake hasn't eaten within the allowed time
                break
            
            sensory_vector = get_sensory()  # Get sensory information from the environment
            activation = net.activate(sensory_vector)  # Process sensory inputs through the network to get outputs
            action = np.argmax(activation)  # Choose the action corresponding to the highest output neuron
            change_direction(action)  # Change the snake's direction based on chosen action
            apple = step()  # Move the snake one step in the game, check for apple eating
            t += 1  # Increment the time step

            if apple:  # If an apple was eaten during this step
                last_ate_apple = t  # Update the last time apple was eaten

        scores.append(len(snake))  # Append the length of the snake (score) to the list

    return np.mean(scores)  # Return the average score over all iterations


def feed_forward_layers(inputs, outputs, connections, genome):
  """
  Modify neat-python's function to display more hidden nodes 
  """
  required = set(genome.nodes)

  layers = []
  s = set(inputs)
  while 1:
      # Find candidate nodes c for the next layer.  These nodes should connect
      # a node in s to a node not in s.
      c = set(b for (a, b) in connections if a in s and b not in s)
      # Keep only the used nodes whose entire input set is contained in s.
      t = set()
      for n in c:
          if n in required and all(a in s for (a, b) in connections if b == n):
              t.add(n)

      if not t:
          break

      layers.append(t)
      s = s.union(t)

  return layers

def modify_eval_functions(net, genome, config):
  """
  Modify neat-python's function to display more hidden nodes 
  """
  # Gather expressed connections.
  connections = [cg.key for cg in genome.connections.values() if cg.enabled]

  layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections, genome)
  node_evals = []
  for layer in layers:
      for node in layer:
          inputs = []
          for conn_key in connections:
              inode, onode = conn_key
              if onode == node:
                  cg = genome.connections[conn_key]
                  inputs.append((inode, cg.weight))

          ng = genome.nodes[node]
          aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
          activation_function = config.genome_config.activation_defs.get(ng.activation)
          node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))
  
  net.node_evals = node_evals

def simulate_animation(net, genome, config):
  global screen
  global font
  reset()

  modify_eval_functions(net, genome, config)
  has_eval = set(eval[0] for eval in net.node_evals)

  has_input = set(con[1] for con in genome.connections)

  hidden_nodes = [node for node in genome.nodes if not 0 <= node <= 3 and node in has_input and node in has_eval]

  node_centers = get_node_centers(net, genome, hidden_nodes)

  last_ate_apple = 0
  screen = pygame.display.set_mode((screenWidth, screenHeight))
  STEP = pygame.USEREVENT + 1
  pygame.time.set_timer(STEP, interval)

  pygame.init()
  font = pygame.font.Font("cmunbtl.otf", 24)
  running = True
  ts = 0
  while running:
    if dead:
      running = False
    if ts - last_ate_apple > MIN_TIME_TO_EAT_APPLE:
      running = False
    
    for event in pygame.event.get():
      if (event.type == pygame.QUIT):
        running = False
      elif (event.type == STEP):
        sensory_vector = get_sensory()
        activation = net.activate(sensory_vector)
        action = np.argmax(activation)
        change_direction(action)
        apple = step()
        if apple:
          last_ate_apple = ts

    screen.fill(BLACK)
    draw_square() 
    draw_snake() 
    draw_apple() 
    draw_network(net, genome, node_centers, hidden_nodes)
    pygame.display.flip()
  pygame.quit()

def get_node_centers(net, genome, hidden_nodes):
  
  node_centers = {}

  startY = window_buffer + NODE_SIZE
  startX = window_buffer

  for i, input_node in enumerate(net.input_nodes):
    center2 = startX + 5.5 * NODE_SIZE, startY + i * 3 * NODE_SIZE + 10
    node_centers[input_node] = center2

  startY = window_buffer + NODE_SIZE
  startX = window_buffer

  startX = window_buffer + 0.5 * networkWidth
  startY = window_buffer + NODE_SIZE * 6

  for i, hidden_node in enumerate(hidden_nodes):
    x = startX + 2 * NODE_SIZE if i % 2 == 0 else startX - 2 * NODE_SIZE
    if i == 2: x += NODE_SIZE * 3
    center2 = x, startY + i * 5 * NODE_SIZE + 10
    node_centers[hidden_node] = center2


  startY = window_buffer + 12 * NODE_SIZE
  startX = screenWidth - gameWidth - window_buffer * 3 - NODE_SIZE

  for i, output_node in enumerate(net.output_nodes):
    center2 = startX - 2 * NODE_SIZE, startY + i * 3 * NODE_SIZE + 10
    node_centers[output_node] = center2

  return node_centers

def draw_connections(first_set, second_set, net, genome, node_centers):
  for first in first_set:
    for second in second_set:
      if (first, second) in genome.connections:
        start = node_centers[first]
        stop = node_centers[second]
        weight = genome.connections[(first, second)].weight
        color = BLUE if weight >= 0 else ORANGE

        surf = pygame.Surface((screenWidth, screenHeight), pygame.SRCALPHA)
        alpha = 255 * (0.3 + net.values[first] * 0.7)
        pygame.draw.line(surf, color + (alpha,), start, stop, width=5)
        screen.blit(surf, (0, 0))

def draw_network(net, genome, node_centers, hidden_nodes):
  node_names = { 
      -1 : "Wall_N",
      -2 : "Wall_S",
      -3 : "Wall_E",
      -4 : "Wall_W",
      -5 : "Tail_N",
      -6 : "Tail_S",
      -7 : "Tail_E",
      -8 : "Tail_W",
      -9 : "Apple_N",
      -10 : "Apple_S",
      -11 : "Apple_E",
      -12 : "Apple_W",
      -13 : "Wall_NE", 
      -14 : "Wall_SE", 
      -15 : "Wall_SW", 
      -16 : "Wall_NW",
      -17 : "Apple_NE",
      -18 : "Apple_SE",
      -19 : "Apple_SW",
      -20 : "Apple_NW", 
      0: 'Up', 1 : "Left", 2 : "Down", 3 : "Right"
  }

  # draw connections between input and output nodes
  draw_connections(net.input_nodes, net.output_nodes, net, genome, node_centers)
  draw_connections(net.input_nodes, hidden_nodes, net, genome, node_centers)
  draw_connections(hidden_nodes, hidden_nodes, net, genome, node_centers)
  draw_connections(hidden_nodes, net.output_nodes, net, genome, node_centers)

  # draw input nodes
  for i, input_node in enumerate(net.input_nodes):
    center = node_centers[input_node]

    center2 = center[0] - 5.5 * NODE_SIZE, center[1] - 10
    img = font.render(node_names[input_node], True, WHITE)
    screen.blit(img, center2)

    color = (net.values[input_node] * 255, 0, 0)

    pygame.draw.circle(screen, color, center, NODE_SIZE)
    pygame.draw.circle(screen, WHITE, center, NODE_SIZE, width=5)

  # draw output nodes
  for i, output_node in enumerate(net.output_nodes):
    center = node_centers[output_node]
    color = (net.values[output_node] * 255, 0, 0)
    pygame.draw.circle(screen, color, center, NODE_SIZE)
    pygame.draw.circle(screen, WHITE, center, NODE_SIZE, width=5)

    center2 = center[0] + 1.5 * NODE_SIZE, center[1] - 10
    img = font.render(node_names[output_node], True, WHITE)
    screen.blit(img, center2)

  # draw hidden nodes

  for hidden in hidden_nodes:
    center = node_centers[hidden]
    color = (net.values[hidden] * 255, 0, 0)

    # center2 = center[0] - 5.5 * NODE_SIZE, center[1] - 10
    # img = font.render(str(hidden), True, WHITE)
    # screen.blit(img, center2)

    pygame.draw.circle(screen, color, center, NODE_SIZE)
    pygame.draw.circle(screen, WHITE, center, NODE_SIZE, width=5)

def draw_snake():
  for i, (x, y) in enumerate(snake):
      rect = pygame.Rect(getLeftTop(x, y), (blockWidth - BUFFER * 2, blockHeight - BUFFER * 2))
      pygame.draw.rect(screen, YELLOW if i == len(snake) - 1 else WHITE, rect)



def get_sensory():
  ''' In this function the snake receives the sensory input. '''

  # Snake's head position
  x, y = snake[-1]

  # Inverted distance to wall for cardinal directions
  dist_to_wall = [
      1 / (y + 1),        # North
      1 / (numRows - y),  # South
      1 / (numCols - x),  # East
      1 / (x + 1)         # West
  ]

  # TODO: add the obstacle and make the distances to that as well -- 4 + 4

  # Inverted distance to wall for diagonal directions
  dist_to_wall_diagonal = [
      1 / (min(numCols - x, y + 1)),       # NE
      1 / (min(numCols - x, numRows - y)), # SE
      1 / (min(x + 1, numRows - y)),       # SW
      1 / (min(x + 1, y + 1))              # NW
  ]

  # Flag for if will hit tail in this cardinal direction
  will_hit_tail = [0, 0, 0, 0]
  for (body_x, body_y) in snake[:-1]:
      if body_x == x:
          if body_y > y:
              will_hit_tail[1] = 1  # South
          else:
              will_hit_tail[0] = 1  # North
      elif body_y == y:
          if body_x > x:
              will_hit_tail[2] = 1  # East
          else:
              will_hit_tail[3] = 1  # West

  a_x, a_y = apple
  
  # Apple in cardinal directions
  apple_info = [
      x == a_x and a_y < y,  # North
      x == a_x and a_y > y,  # South
      y == a_y and a_x > x,  # East
      y == a_y and a_x < x,  # West
  ]

  # Apple in diagonal directions
  apple_info_diagonal = [
      (x < a_x and y > a_y),  # NE
      (x < a_x and y < a_y),  # SE
      (x > a_x and y < a_y),  # SW
      (x > a_x and y > a_y)   # NW
  ]

  # Combine all sensory information into one array
  sensory_vector = np.array(dist_to_wall + dist_to_wall_diagonal + will_hit_tail + apple_info + apple_info_diagonal)

  return 1.0 * sensory_vector
# returns the 12 item array adding all of the 3 elements * 4 directions // plus the 4 + 4 with diagonal 

def change_direction(code):
  global v_x
  global v_y
  assert(0 <= code <= 3)

  # wasd
  if code == 0:
    v_x = 0
    v_y = -1
  elif code == 1:
    v_x = -1
    v_y = 0
  elif code == 2:
    v_x = 0
    v_y = 1
  else:
    v_x = 1
    v_y = 0 

def step():
  global apple
  global dead

  ate_apple = False

  x, y = snake[-1]
  snake.append((x + v_x, y + v_y))
  x, y = snake[-1]

  # hit wall
  if x < 0 or x >= numCols or y < 0 or y >= numRows:
    dead = True

  # hit body
  for s in snake[:-1]:
    if s == snake[-1]:
      dead = True
      break

  if not snake[-1] == apple:
    snake.pop(0)
  else:
    apple = (int) (random() * numCols), (int) (random() * numRows)
    ate_apple = True

  return ate_apple

def draw_square():
  draw = gameTopLeft[0] - BUFFER, gameTopLeft[1] - BUFFER
  rect = pygame.Rect(draw, (gameWidth + 2 * BUFFER, gameHeight + 2 * BUFFER))
  pygame.draw.rect(screen, WHITE, rect, width=BUFFER // 2)

def getLeftTop(x, y):
    return (x / numRows) * gameWidth + BUFFER + gameTopLeft[0], (y / numRows) * gameHeight + BUFFER + gameTopLeft[1]

def draw_apple():
  x, y = apple
  rect = pygame.Rect(getLeftTop(x, y), (blockWidth - BUFFER * 2, blockHeight - BUFFER * 2))
  pygame.draw.rect(screen, RED, rect)

if __name__ == "__main__":
  pass