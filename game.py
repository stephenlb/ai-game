# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Callable

import pygame
import torch
import numpy as np

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
    'Kamala',
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
    'GOD',
]

@dataclass
class Entity:
    name: str
    position: pygame.Vector2
    speed: float
    size: int
    color: str

@dataclass
class AI(Entity):
    ### TODO: circular dep problem
    #render: Callable
    level: int
    model: None
    learning: float
    optimizer: torch.optim.adamw.AdamW
    loss: torch.nn.modules.loss.MSELoss
    losses: List

@dataclass
class Player(Entity):
    happiness: float

## Main Game State
@dataclass
class Game:
    player: Player
    ai: AI
    running: bool
    width: int
    height: int
    state: str # intro, game over, play
    delta: int
    screen: pygame.surface.Surface
    clock: pygame.time.Clock

## Main Game Setup
width  = 720
height = 1280
player = Player(
    happiness=100.0,
    name=np.random.choice(PLAYER_NAMES),
    position=pygame.Vector2(1, 1),
    speed=1,
    size=60,
    color="green",
)
model = torch.nn.Sequential(
    torch.nn.Linear(4, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 2),
    torch.nn.Tanh(),
)
ai = AI(
    name ="AI",
    position=pygame.Vector2(360, 640),
    level=1,
    speed=1,
    size=120,
    color="red",
    model=model,
    learning=1e-3,
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
    loss=torch.nn.MSELoss(),
    losses=[],
)
game = Game(
    player=player,
    ai=ai,
    width=width,
    height=height,
    running=True,
    state="play",
    delta=0,
    screen=pygame.display.set_mode((width, height)),
    clock=pygame.time.Clock(),
)

## Render Player and AI
def render_entity(game: Game, entity: Entity):
    pygame.draw.circle(game.screen, entity.color, entity.position, entity.size)

def render_player(game: Game):
    x, y = pygame.mouse.get_pos()
    game.player.position.x = x
    game.player.position.y = y
    render_entity(game, game.player)

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

    game.ai.position.x += (width/2)  * ai_directions[0][0] * game.delta
    game.ai.position.y += (height/2) * ai_directions[0][1] * game.delta

    render_entity(game, game.ai)

def render_game(game: Game):
    game.screen.fill("purple")

    render_player(game)
    render_ai(game)

    pygame.display.flip()
    game.delta = game.clock.tick(60) / 1000

def render_game_over(game: Game):
    game.screen.fill("black")
    font = pygame.font.Font(None, 74)
    text = font.render("Game Over", True, "red")
    game.screen.blit(text, (game.width // 2 - text.get_width() // 2, game.height // 2 - text.get_height() // 2))
    pygame.display.flip()
    pygame.time.delay(3000)
    game.running = False

def train(game: Game, features, labels):
    ## TODO consider encoding/decoding inside this function
    ## TODO consider encoding/decoding inside this function
    ## TODO consider encoding/decoding inside this function
    features = torch.tensor(features, dtype=torch.float32)
    labels   = torch.tensor(labels, dtype=torch.float32)
    game.ai.optimizer.zero_grad()
    output = game.ai.model(features)

    ## TODO based on level
    if len(game.ai.losses) > game.ai.level * 100:
        return output

    print(output)

    loss = game.ai.loss(output, labels)
    game.ai.losses.append(loss.item())
    game.ai.losses = game.ai.losses[-500:]

    loss.backward()
    game.ai.optimizer.step()

    print(f"Loss ({len(game.ai.losses)}:", sum(game.ai.losses) / len(game.ai.losses))

    return output

## Main Game Loop
pygame.init()
while game.running:
    ## Handle Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ## Game State Management
    match game.state:
        case "intro":
            pass

        case "play":
            render_game(game)

        case "game_over":
            render_game_over(game)

pygame.quit()
