import sys
import logging
import itertools
import inspect
from typing import Optional, Dict
from copy import deepcopy
import numpy as np
import pygame
from enum import Enum


class MoveType(Enum):
    UP = "w"
    LEFT = "a"
    DOWN = "s"
    RIGHT = "d"


class MissingParameterError(Exception):
    pass


def handle_inplace(
    *, using_array: Optional[str] = None, validate: Optional[Dict] = None
):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            sig = inspect.signature(func)
            required_params = ["inplace"]
            if using_array is not None:
                required_params.append(using_array)
            for param in required_params:
                if param not in sig.parameters:
                    self._throw(
                        message=f"`{func.__name__}` must have `{param}` as a keyword parameter",
                        error_type=MissingParameterError,
                    )
            inplace = kwargs.get("inplace", sig.parameters["inplace"].default)
            default_validate = validate if validate is not None else {}
            if using_array is not None:
                self._check_is_ndarray(
                    kwargs[using_array], ndim=default_validate.get("ndim")
                )
                kwargs[using_array] = (
                    kwargs[using_array] if inplace else np.copy(using_array)
                )
            else:
                self._check_is_ndarray(self.tiles, ndim=default_validate.get("ndim"))
                self = self if inplace else deepcopy(self)
            func(self, *args, **kwargs)
            return self.tiles if using_array is None else kwargs[using_array]

        return wrapper

    return decorator


class TileGame:
    """ """

    def __init__(self, size=4):
        self.tiles = np.zeros((size, size), dtype=np.int64)
        self.size = size
        self.score = 0
        self._logger = logging.getLogger("2048")
        self._is_modified = False

    def _throw(self, *, message, error_type=Exception):
        response = f"{sys._getframe(1).f_code.co_name}: {message}"
        self._logger.error(response)
        raise error_type(response)

    def _check_is_ndarray(self, tiles, *, ndim=None):
        if not isinstance(tiles, np.ndarray):
            self._throw(
                message=f"{type(tiles).__name__=} != np.ndarray",
                error_type=TypeError,
            )
        if ndim is not None:
            if not isinstance(ndim, int) and ndim > 0:
                self._throw(
                    message=f"{type(ndim).__name__=} != int or {ndim=} > 0",
                    error_type=TypeError,
                )
            elif ndim != tiles.ndim:
                self._throw(
                    message=f"{tiles.ndim=} != {ndim=}", error_type=AttributeError
                )

    def _check_is_collapsed(self, tiles):
        self._check_is_ndarray(tiles)
        zero_valued_tiles = np.where(tiles == 0)[0]
        if not zero_valued_tiles.size:
            return True
        z = zero_valued_tiles[0]
        return np.all(tiles[z:] == 0)

    def is_winning(self):
        return np.max(self.tiles) >= 2048

    def is_playable(self):
        def has_matching_neighbors():
            return any(
                0 <= x + dx < self.size
                and 0 <= y + dy < self.size
                and self.tiles[x, y] == self.tiles[x + dx, y + dy]
                for x, y in itertools.product(range(self.size), repeat=2)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            )

        return np.any(self.tiles == 0) or has_matching_neighbors()

    @handle_inplace(validate={"ndim": 2})
    def insert_random_tiles(self, *, size=1, filter=None, inplace=True):
        if not (1 <= size <= self.tiles.size):
            self._throw(
                message=f"1 <= {size=} <= {self.tiles.size=}", error_type=ValueError
            )
        if filter is not None:
            zero_indices = np.argwhere(filter)
            # NOTE: Happens when `zero_indices < size < self.tiles.size`
            if size > len(zero_indices):
                self._throw(
                    message=f"{size=} > {len(zero_indices)=}",
                    error_type=ValueError,
                )
            else:
                random_tile_indices = zero_indices[
                    np.random.choice(zero_indices.shape[0], size=size, replace=False),
                    :,
                ]
        else:
            flat_indices = np.random.choice(self.tiles.size, size=size, replace=False)
            random_tile_indices = np.column_stack(
                np.unravel_index(flat_indices, self.tiles.shape)
            )
        self.tiles[tuple(random_tile_indices.T)] = 4 if np.random.uniform() < 0.1 else 2

    @handle_inplace(validate={"ndim": 2})
    def collapse_aligned_tiles_2d(self, move, *, inplace=False):
        aligned_tiles = self.align_tiles_by(move)
        non_zero_indices = np.argwhere(aligned_tiles != 0)
        original_non_zero_entries = np.argwhere(aligned_tiles != 0)
        for i in np.unique(non_zero_indices[:, 0]):
            indices = np.argwhere(non_zero_indices[:, 0] == i).flatten()
            non_zero_indices[indices, 1] = np.arange(len(indices))
        mask = np.zeros_like(aligned_tiles, dtype=bool)
        mask[tuple(non_zero_indices.T)] = True
        aligned_tiles[mask] = aligned_tiles[tuple(original_non_zero_entries.T)]
        aligned_tiles[~mask] = 0

    @handle_inplace(using_array="view", validate={"ndim": 1})
    def merge_aligned_tiles_1d(self, *, view, inplace=False):
        self._check_is_collapsed(view)
        for i in range(len(view) - 1):
            if view[i] and view[i] == view[i + 1]:
                view[i] *= 2
                view[i + 1] = 0
                self.score += view[i]
                self._is_modified = True

    def align_tiles_by(self, move):
        self._check_is_ndarray(self.tiles, ndim=2)
        if move == MoveType.UP:
            self._logger.debug(f"Aligning Tiles: {np.rot90(self.tiles).tolist()=}")
            return np.rot90(self.tiles)
        elif move == MoveType.LEFT:
            self._logger.debug(f"Aligning Tiles: {self.tiles.tolist()=}")
            return self.tiles
        elif move == MoveType.DOWN:
            self._logger.debug(
                f"Aligning Tiles: {np.rot90(self.tiles, k=-1).tolist()=}"
            )
            return np.rot90(self.tiles, k=-1)
        elif move == MoveType.RIGHT:
            self._logger.debug(f"Aligning Tiles: {np.fliplr(self.tiles).tolist()=}")
            return np.fliplr(self.tiles)
        else:
            self._throw(
                message=f"{move=} not in {list(MoveType)}", error_type=ValueError
            )

    def update(self, move):
        collapsed_tiles = self.collapse_aligned_tiles_2d(move)
        self._is_modified = np.any(self.tiles != collapsed_tiles)
        self.tiles = collapsed_tiles

        aligned_tiles = self.align_tiles_by(move)
        np.apply_along_axis(
            lambda view: self.merge_aligned_tiles_1d(view=view, inplace=True),
            axis=1,
            arr=aligned_tiles,
        )
        self.collapse_aligned_tiles_2d(move, inplace=True)
        if self._is_modified:
            self.insert_random_tiles(filter=self.tiles == 0)


pygame.init()
pygame.display.set_caption("2048")
screen = pygame.display.set_mode((400, 400))
font = pygame.font.Font("freesansbold.ttf", 24)
timer = pygame.time.Clock()
fps = 60

colors = {
    0: (204, 192, 179),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    "light text": (249, 246, 242),
    "dark text": (119, 110, 101),
    "other": (0, 0, 0),
    "bg": (187, 173, 160),
}


def draw(board):
    pygame.draw.rect(screen, colors["bg"], [0, 0, 400, 400], 0, 0)
    for i, j in itertools.product(range(board.size), repeat=2):
        value = board.tiles[i, j]
        if value > 8:
            value_color = colors["light text"]
        else:
            value_color = colors["dark text"]
        if value <= 2048:
            color = colors[value]
        else:
            color = colors["other"]
        pygame.draw.rect(screen, color, [j * 95 + 20, i * 95 + 20, 75, 75], 0, 5)
        if value > 0:
            value_len = len(str(value))
            font = pygame.font.Font("freesansbold.ttf", 48 - (5 * value_len))
            value_text = font.render(str(value), True, value_color)
            text_rect = value_text.get_rect(center=(j * 95 + 57, i * 95 + 57))
            screen.blit(value_text, text_rect)
            pygame.draw.rect(screen, "black", [j * 95 + 20, i * 95 + 20, 75, 75], 2, 5)


def main():
    board = TileGame()
    board.insert_random_tiles(size=2, inplace=True)
    endless = False
    playable = False
    while True:
        timer.tick(fps)
        draw(board)
        if board.is_winning() and not board.is_playable():
            raise NotImplementedError(f"{board.tiles=}")
        elif not endless and board.is_winning():
            pygame.draw.rect(screen, "black", [50, 50, 300, 100], 0, 10)
            game_over_text1 = font.render("Congrats!", True, "green")
            game_over_text2 = font.render("Press Enter to Continue", True, "green")
            screen.blit(game_over_text1, (130, 65))
            screen.blit(game_over_text2, (70, 105))
        elif not board.is_playable():
            playable = True
            pygame.draw.rect(screen, "black", [50, 50, 300, 100], 0, 10)
            game_over_text1 = font.render("Game Over!", True, "red")
            game_over_text2 = font.render("Press Enter to Restart", True, "red")
            screen.blit(game_over_text1, (130, 65))
            screen.blit(game_over_text2, (70, 105))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    board.update(MoveType.UP)
                elif event.key == pygame.K_a:
                    board.update(MoveType.LEFT)
                elif event.key == pygame.K_s:
                    board.update(MoveType.DOWN)
                elif event.key == pygame.K_d:
                    board.update(MoveType.RIGHT)
                elif event.key == pygame.K_q:
                    pygame.quit()
                elif event.key == pygame.K_RETURN and not endless:
                    endless = True
                elif event.key == pygame.K_RETURN and playable:
                    playable = False
                    board.tiles = np.zeros_like(board.tiles)
                    board.insert_random_tiles(size=2)
        pygame.display.flip()


if __name__ == "__main__":
    main()
