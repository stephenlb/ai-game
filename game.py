# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Callable

import pygame
import pygame.font
import torch
import numpy as np

pygame.init()
pygame.font.get_init()
pygame.mixer.init()
pygame.mixer.music.load("ai.wav")
pygame.mixer.music.set_volume(0.01)
pygame.mixer.music.play(-1)

## TODO: SCARY
## TODO: as AI gets closer the screen shakes, gets red
## TODO: scary music gets louder as AI gets closer
## TODO: 
## TODO: STORY?
## TODO: why is the AI chasing the player? THOR AND HAMMER?
## TODO: 
## TODO: Max level for winner!!!! 20 levels
## TODO: 
## TODO: 
## TODO: Ideas
## TODO: when you player is captured, next player can be saved
## TODO: 
## TODO: 

## TODO: make AI predict ahead where the player is going
## TODO: list of names for the plary
## TODO: add places to hide
## TODO: name the game
## TODO: make player grow as game difficulty increases
## TODO: make player slow as game difficulty increases

## Game Name Ideas
## -------------------
## AI Terminator
## ChAIse
## Jury Duty 5
## Revenge of The AI IRS
## Stalker
## Wanted
## You Got AI Mail
## City of Dis
##


PLAYER_NAMES = [
    'QuantifiedQuantum',
    'Kalamata',
    'Torva',
    'BoboBear',
    'Kevin',
    'Barry',
    'Kyle',
    'Tuleku',
    'Travis',
    'Valor',
    'Lukey',
    'Mosh',
    'Alazr',
    'Ahmed',
]

@dataclass
class Entity:
    name: str
    position: pygame.Vector2
    speed: float
    size: int
    color: Tuple[int, int, int]
    text_color: Tuple[int, int, int]

@dataclass
class AI(Entity):
    level: int
    model: None
    learning: int
    optimizer: torch.optim.adamw.AdamW
    loss: torch.nn.modules.loss.MSELoss
    knowledge: int

@dataclass
class Player(Entity):
    happiness: float

@dataclass
class Background():
    color: Tuple[int, int, int]
    shake: int
    rect: pygame.Rect

## Main Game State
@dataclass
class Game:
    player: Player
    ai: AI
    background: Background
    running: bool
    width: int
    height: int
    state: str # intro, game over, play
    delta: int
    frame: int
    game_over_frame: int
    font: pygame.font.Font
    screen: pygame.surface.Surface
    clock: pygame.time.Clock

## Game Resolution
width  = 720
height = 1280

## Main Game Setup
player = Player(
    happiness=100.0,
    name=np.random.choice(PLAYER_NAMES),
    position=pygame.Vector2(1, 1),
    speed=0,
    size=60,
    color=(0, 255, 255),
    text_color=(0, 155, 155),
)
## TODO: reduce modle for lower levels
model = torch.nn.Sequential(
    torch.nn.Linear(4, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 2),
    torch.nn.Tanh(),
)
ai = AI(
    name ="AI",
    position=pygame.Vector2(360, 640),
    level=1,
    speed=1,
    size=120,
    color=(255, 0, 0),
    text_color=(255, 255, 255),
    model=model,
    learning=5, # 1 is highest, 10 is lowest
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
    loss=torch.nn.MSELoss(),
    knowledge=0,
)
game = Game(
    player=player,
    ai=ai,
    background=Background(
        color=(100, 0, 250),
        shake=0,
        rect=pygame.Rect(0,0,width,height)
    ),
    font=pygame.font.SysFont("Arial", 26),
    width=width,
    height=height,
    running=True,
    state="play",
    delta=0,
    frame=0,
    game_over_frame=0,
    screen=pygame.display.set_mode((width, height)),
    clock=pygame.time.Clock(),
)

def set_level(game: Game, level: int):
    game.ai.level      = level
    game.ai.knowledge  = 0
    game.ai.speed      = 1 + (level * 0.4)
    game.ai.size       = 120 + (level * 10)
    game.ai.position.x = 360
    game.ai.position.y = -game.ai.size

def collision(game: Game):
    distance = proximity(
        ai.position.x,
        ai.position.y,
        player.position.x,
        player.position.y,
    )
    radi = ai.size + player.size
    if distance <= radi:
        return True, distance - radi
    else:
        return False, distance - radi

def proximity(x1, y1, x2, y2):
    xd = (x1 - x2) ** 2
    yd = (y1 - y2) ** 2
    return int(np.sqrt(xd + yd))

def render_game_scene(game: Game):
    render_background(game)
    render_player(game)
    render_ai(game)

    pygame.display.flip()
    game.delta = game.clock.tick(60) / 1000

def render_game_over_scene(game: Game):
    game.screen.fill("black")
    text = game.font.render("Game Over", True, "red")
    game.screen.blit(text, (game.width // 2 - text.get_width() // 2, game.height // 2 - text.get_height() // 2))
    pygame.display.flip()

    ## Restart Game in a few seconds
    if game.frame > game.game_over_frame + 600:
        game.state = "play"
        set_level(game, level=1)

## Render Player and AI
def render_entity(game: Game, entity: Entity):
    position = entity.position
    pygame.draw.circle(game.screen, entity.color, entity.position, entity.size)

    text = game.font.render(entity.name, True, entity.text_color)
    game.screen.blit(text, (
        position.x - text.get_width()  // 2,
        position.y - text.get_height() // 2,
    ))

def render_player(game: Game):
    x, y = pygame.mouse.get_pos()
    position = game.player.position
    position.x = x
    position.y = y

    collided, distance = collision(game)
    if 400 >= distance:
        game.background.shake = (400 - distance) // 50
        pygame.mixer.music.set_volume((400 - distance)/400)
    else:
        game.background.shake = 0
        pygame.mixer.music.set_volume(0.01)

    print("shake",game.background.shake)

    shake = game.background.shake
    if shake:
        position.x += np.random.randint(-shake, shake)
        position.y += np.random.randint(-shake, shake)

    render_entity(game, game.player)

    collided, distance = collision(game)
    if collided:
        game.game_over_frame = game.frame
        game.state = "game over"

def render_ai(game: Game):
    ## Calculate slope for both x and y for AI to Player
    dx = game.player.position.x - game.ai.position.x
    dy = game.player.position.y - game.ai.position.y

    ## Normalize as well
    width  = game.width
    height = game.height
    labels   = [[dx/width, dy/height]]
    features = [[
        game.player.position.x / width,
        game.player.position.y / height,
        game.ai.position.x     / width,
        game.ai.position.y     / height,
    ]]

    ## Train and get latest directions
    ai_directions = train(game, features, labels)

    game.ai.position.x += (width/2)  * ai_directions[0][0] * game.delta * game.ai.speed
    game.ai.position.y += (height/2) * ai_directions[0][1] * game.delta * game.ai.speed

    render_entity(game, game.ai)

def render_background(game: Game):
    game.screen.fill("black")
    shake = game.background.shake
    if shake:
        game.background.rect.x = np.random.randint(0, shake)
        game.background.rect.y = np.random.randint(0, shake)
        game.background.rect.width = game.width - np.random.randint(0, shake)
        game.background.rect.height = game.height - np.random.randint(0, shake)
    else:
        game.background.rect.x = 0
        game.background.rect.y = 0
        game.background.rect.width = game.width
        game.background.rect.height = game.height
    
    pygame.draw.rect(game.screen, game.background.color, game.background.rect)

def train(game: Game, features, labels):
    game.ai.optimizer.zero_grad()

    features = torch.tensor(features, dtype=torch.float32)
    labels   = torch.tensor(labels, dtype=torch.float32)
    output   = game.ai.model(features)

    game.ai.knowledge += 1
    if game.ai.knowledge > game.ai.level * 150:
        set_level(game, game.ai.level + 1)
        return output

    ## TODO based on level?
    ## Train once every 5 frames
    if game.frame % game.ai.learning > 0:
        return output

    loss = game.ai.loss(output, labels)
    loss.backward()
    game.ai.optimizer.step()

    ## TODO REMOVE
    #game.ai.losses.append(loss.item())
    #game.ai.losses = game.ai.losses[-500:]
    #print(
    #    f"{len(game.ai.losses)}:",
    #    sum(game.ai.losses) / len(game.ai.losses)
    #)

    return output

## Main Game Loop
while game.running:
    ## Handle Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game.running = False

    ## Game State Management
    game.frame += 1
    match game.state:
        case "intro":
            pass

        case "play":
            render_game_scene(game)

        case "game over":
            render_game_over_scene(game)

pygame.quit()
