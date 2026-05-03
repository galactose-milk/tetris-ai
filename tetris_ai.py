"""
Tetris AI - Play as Human or watch the AI solve it using Heuristics!
Controls (Human Mode):
  LEFT/RIGHT  - Move piece
  UP          - Rotate piece
  DOWN        - Soft drop
  SPACE       - Hard drop
  P           - Pause
  ESC         - Back to menu
"""

import pygame
import random
import sys
import time
import math

# ─────────────────────── CONSTANTS ────────────────────────────────────────────
BOARD_W, BOARD_H = 10, 20
CELL = 32                          # pixels per cell
PANEL_W = 320                      # right‑side info/heuristic panel
MARGIN = 20

WINDOW_W = MARGIN + BOARD_W * CELL + MARGIN + PANEL_W + MARGIN
WINDOW_H = MARGIN + BOARD_H * CELL + MARGIN

FPS = 60

# ─── Colors ────────────────────────────────────────────────────────────────────
BG_TOP      = (6, 10, 22)
BG_BOTTOM   = (18, 24, 42)
BG          = BG_TOP
GRID_COLOR  = (26, 32, 56)
PANEL_BG    = (14, 18, 32)
PANEL_EDGE  = (72, 88, 132)
WHITE       = (242, 244, 255)
GRAY        = (142, 150, 176)
DARK_GRAY   = (34, 40, 58)
ACCENT      = (0, 200, 255)
ACCENT_2    = (255, 182, 72)
SUCCESS     = (98, 224, 154)
DANGER      = (255, 110, 122)
GHOST_ALPHA = 80

PIECE_COLORS = {
    'I': (0,   220, 220),
    'O': (240, 200,   0),
    'T': (180,  50, 200),
    'S': (50,  200,  50),
    'Z': (220,  50,  50),
    'J': (50,   80, 220),
    'L': (220, 140,  30),
}

# ─────────────────────── TETROMINOES ──────────────────────────────────────────
SHAPES = {
    'I': [
        [(0,1),(1,1),(2,1),(3,1)],
        [(2,0),(2,1),(2,2),(2,3)],
        [(0,2),(1,2),(2,2),(3,2)],
        [(1,0),(1,1),(1,2),(1,3)],
    ],
    'O': [
        [(0,0),(1,0),(0,1),(1,1)],
        [(0,0),(1,0),(0,1),(1,1)],
        [(0,0),(1,0),(0,1),(1,1)],
        [(0,0),(1,0),(0,1),(1,1)],
    ],
    'T': [
        [(1,0),(0,1),(1,1),(2,1)],
        [(1,0),(1,1),(2,1),(1,2)],
        [(0,1),(1,1),(2,1),(1,2)],
        [(1,0),(0,1),(1,1),(1,2)],
    ],
    'S': [
        [(1,0),(2,0),(0,1),(1,1)],
        [(1,0),(1,1),(2,1),(2,2)],
        [(1,1),(2,1),(0,2),(1,2)],
        [(0,0),(0,1),(1,1),(1,2)],
    ],
    'Z': [
        [(0,0),(1,0),(1,1),(2,1)],
        [(2,0),(1,1),(2,1),(1,2)],
        [(0,1),(1,1),(1,2),(2,2)],
        [(1,0),(0,1),(1,1),(0,2)],
    ],
    'J': [
        [(0,0),(0,1),(1,1),(2,1)],
        [(1,0),(2,0),(1,1),(1,2)],
        [(0,1),(1,1),(2,1),(2,2)],
        [(1,0),(1,1),(0,2),(1,2)],
    ],
    'L': [
        [(2,0),(0,1),(1,1),(2,1)],
        [(1,0),(1,1),(1,2),(2,2)],
        [(0,1),(1,1),(2,1),(0,2)],
        [(0,0),(1,0),(1,1),(1,2)],
    ],
}

PIECE_NAMES = list(SHAPES.keys())

# ─────────────────────── HEURISTIC WEIGHTS ────────────────────────────────────
# GA-optimized weights from Yiyuan Lee's research + additional robustness features
# These were computed via genetic algorithm on the unit 3-sphere and extended
# with max_height and well_depth penalties for practical robustness.
DEFAULT_WEIGHTS = {
    'complete_lines':    0.760666,   # reward clearing lines
    'aggregate_height': -0.510066,   # penalize tall stacks
    'holes':            -0.35663,    # penalize buried gaps
    'bumpiness':        -0.184483,   # penalize uneven surfaces
    'max_height':       -0.30,       # penalize tallest column spikes
    'well_depth':       -0.15,       # penalize deep wells
}

CUSTOM_WEIGHT_RANGE = (-1.5, 1.5)


def clamp(value, low, high):
    return max(low, min(high, value))

# ══════════════════════════════════════════════════════════════════════════════
#  BOARD CLASS
# ══════════════════════════════════════════════════════════════════════════════
class Board:
    def __init__(self):
        self.grid = [[None]*BOARD_W for _ in range(BOARD_H)]

    def copy(self):
        b = Board()
        b.grid = [row[:] for row in self.grid]
        return b

    def is_valid(self, cells):
        for (x, y) in cells:
            if x < 0 or x >= BOARD_W or y >= BOARD_H:
                return False
            if y >= 0 and self.grid[y][x] is not None:
                return False
        return True

    def lock(self, cells, color):
        for (x, y) in cells:
            if 0 <= y < BOARD_H:
                self.grid[y][x] = color

    def clear_lines(self):
        new_grid = [row for row in self.grid if any(c is None for c in row)]
        lines = BOARD_H - len(new_grid)
        new_grid = [[None]*BOARD_W for _ in range(lines)] + new_grid
        self.grid = new_grid
        return lines

    # Heuristic computations ─────────────────────────────────────────────────
    def column_heights(self):
        heights = []
        for x in range(BOARD_W):
            h = 0
            for y in range(BOARD_H):
                if self.grid[y][x] is not None:
                    h = BOARD_H - y
                    break
            heights.append(h)
        return heights

    def aggregate_height(self):
        return sum(self.column_heights())

    def max_height(self):
        """Height of the tallest column — penalizes dangerous spike-ups."""
        return max(self.column_heights())

    def count_holes(self):
        holes = 0
        for x in range(BOARD_W):
            found_block = False
            for y in range(BOARD_H):
                if self.grid[y][x] is not None:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def bumpiness(self):
        heights = self.column_heights()
        return sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))

    def well_depth(self):
        """Sum of well depths. A well exists where a column is lower than both
        its neighbors. Edge columns only need one neighbor to be higher."""
        heights = self.column_heights()
        total = 0
        for i in range(BOARD_W):
            if i == 0:
                # left edge — compare to right neighbor only
                well = max(0, heights[1] - heights[0])
            elif i == BOARD_W - 1:
                # right edge — compare to left neighbor only
                well = max(0, heights[BOARD_W - 2] - heights[BOARD_W - 1])
            else:
                # interior column — must be lower than both neighbors
                left_diff = heights[i - 1] - heights[i]
                right_diff = heights[i + 1] - heights[i]
                if left_diff > 0 and right_diff > 0:
                    well = min(left_diff, right_diff)
                else:
                    well = 0
            total += well
        return total

    def evaluate(self, weights=None):
        """Evaluate board state with 6-feature heuristic.
        Returns (score, lines, agg_h, holes, bump, max_h, wells)."""
        weights = weights or DEFAULT_WEIGHTS
        temp_board = self.copy()
        lines  = temp_board.clear_lines()
        agg_h  = temp_board.aggregate_height()
        holes  = temp_board.count_holes()
        bump   = temp_board.bumpiness()
        max_h  = temp_board.max_height()
        wells  = temp_board.well_depth()

        score = (weights['complete_lines']    * lines +
                 weights['aggregate_height']  * agg_h +
                 weights['holes']             * holes +
                 weights['bumpiness']         * bump  +
                 weights['max_height']        * max_h +
                 weights['well_depth']        * wells)
        return score, lines, agg_h, holes, bump, max_h, wells


# ══════════════════════════════════════════════════════════════════════════════
#  PIECE CLASS
# ══════════════════════════════════════════════════════════════════════════════
class Piece:
    def __init__(self, name=None):
        self.name     = name or random.choice(PIECE_NAMES)
        self.color    = PIECE_COLORS[self.name]
        self.rotation = 0
        self.x        = BOARD_W // 2 - 2
        self.y        = -2

    @property
    def cells(self):
        return [(self.x + dx, self.y + dy)
                for (dx, dy) in SHAPES[self.name][self.rotation]]

    def rotated(self, r):
        return [(self.x + dx, self.y + dy)
                for (dx, dy) in SHAPES[self.name][r % 4]]

    def moved(self, dx=0, dy=0):
        cells = []
        for (cx, cy) in self.cells:
            cells.append((cx+dx, cy+dy))
        return cells

class WeightSlider:
    def __init__(self, key, label, color, value, min_value=None, max_value=None):
        self.key = key
        self.label = label
        self.color = color
        self.value = value
        self.min_value = CUSTOM_WEIGHT_RANGE[0] if min_value is None else min_value
        self.max_value = CUSTOM_WEIGHT_RANGE[1] if max_value is None else max_value
        self.dragging = False
        self.row_rect = pygame.Rect(0, 0, 0, 0)
        self.track_rect = pygame.Rect(0, 0, 0, 0)

    def layout(self, x, y, width):
        self.row_rect = pygame.Rect(x, y, width, 28)
        label_w = 96
        value_w = 60
        track_x = x + label_w
        track_w = max(20, width - label_w - value_w - 12)
        self.track_rect = pygame.Rect(track_x, y + 11, track_w, 6)

    def _value_from_x(self, x):
        if self.track_rect.width <= 0:
            return self.value
        ratio = clamp((x - self.track_rect.x) / self.track_rect.width, 0.0, 1.0)
        return self.min_value + ratio * (self.max_value - self.min_value)

    def set_value(self, value):
        self.value = clamp(value, self.min_value, self.max_value)

    def handle_event(self, event):
        changed = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.row_rect.collidepoint(event.pos):
                self.dragging = True
                self.set_value(self._value_from_x(event.pos[0]))
                changed = True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.set_value(self._value_from_x(event.pos[0]))
            changed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        return changed

    def draw(self, screen, font, value_font):
        label_surf = font.render(self.label, True, WHITE)
        screen.blit(label_surf, (self.row_rect.x, self.row_rect.y + 4))

        pygame.draw.rect(screen, DARK_GRAY, self.track_rect, border_radius=4)
        ratio = (self.value - self.min_value) / max(self.max_value - self.min_value, 1e-9)
        fill_w = int(self.track_rect.width * clamp(ratio, 0.0, 1.0))
        if fill_w > 0:
            pygame.draw.rect(screen, self.color,
                             (self.track_rect.x, self.track_rect.y, fill_w, self.track_rect.height),
                             border_radius=4)

        knob_x = self.track_rect.x + fill_w
        knob_rect = pygame.Rect(0, 0, 12, 18)
        knob_rect.center = (knob_x, self.track_rect.centery)
        pygame.draw.rect(screen, WHITE, knob_rect, border_radius=4)
        pygame.draw.rect(screen, self.color, knob_rect, 2, border_radius=4)

        value_surf = value_font.render(f"{self.value:+.3f}", True, self.color)
        screen.blit(value_surf, (self.row_rect.right - value_surf.get_width(), self.row_rect.y + 5))


# ══════════════════════════════════════════════════════════════════════════════
#  AI SOLVER
# ══════════════════════════════════════════════════════════════════════════════
class AISolver:
    def __init__(self, weights=None):
        self.weights = weights or dict(DEFAULT_WEIGHTS)

    @staticmethod
    def _hard_drop_piece(board, piece_name, rot, col):
        """Simulate dropping a piece from the top. Returns (cells, valid)."""
        sy = -2
        last_valid_y = sy
        valid_start = False
        while True:
            cells = [(col + dx, sy + 1 + dy)
                     for (dx, dy) in SHAPES[piece_name][rot]]
            if board.is_valid(cells):
                sy += 1
                last_valid_y = sy
                valid_start = True
            else:
                break

        final_cells = [(col + dx, last_valid_y + dy)
                       for (dx, dy) in SHAPES[piece_name][rot]]

        # Must be a valid resting position with all cells on the board
        if not valid_start:
            return final_cells, False
        if not board.is_valid(final_cells):
            return final_cells, False
        if not all(0 <= x < BOARD_W for x, y in final_cells):
            return final_cells, False
        if not any(0 <= y < BOARD_H for x, y in final_cells):
            return final_cells, False

        return final_cells, True

    def _score_placement(self, board, piece_name, piece_color, rot, col):
        """Score a single placement. Returns (score, info) or None if invalid."""
        cells, valid = self._hard_drop_piece(board, piece_name, rot, col)
        if not valid:
            return None

        sim_board = board.copy()
        sim_board.lock(cells, piece_color)
        score, lines, h, holes, bump, max_h, wells = sim_board.evaluate(self.weights)

        info = {
            'score':      score,
            'lines':      lines,
            'height':     h,
            'holes':      holes,
            'bumpiness':  bump,
            'max_height': max_h,
            'well_depth': wells,
            'rotation':   rot,
            'column':     col,
        }
        return score, info

    def find_best_move(self, board: 'Board', piece: 'Piece',
                       next_piece: 'Piece' = None):
        """Try all rotations × columns with conditional next-piece lookahead.
        Lookahead activates when the board is getting dangerous (max_height > 10).
        Returns (rotation, x, heuristic_info)."""
        best_score = float('-inf')
        best       = (0, piece.x)
        best_info  = {}

        # Only do lookahead when board is getting dangerous
        max_h = board.max_height()
        use_lookahead = (next_piece is not None and max_h > 10)

        # O-piece has identical rotations, skip duplicates
        num_rots = 1 if piece.name == 'O' else 4

        for rot in range(num_rots):
            for col in range(-2, BOARD_W):
                result = self._score_placement(
                    board, piece.name, piece.color, rot, col)
                if result is None:
                    continue

                base_score, info = result

                # ── Conditional next-piece lookahead ─────────────────────
                if use_lookahead:
                    cells, _ = self._hard_drop_piece(board, piece.name, rot, col)
                    lookahead_board = board.copy()
                    lookahead_board.lock(cells, piece.color)
                    lookahead_board.clear_lines()

                    best_next_score = float('-inf')
                    n_rots = 1 if next_piece.name == 'O' else 4
                    for n_rot in range(n_rots):
                        for n_col in range(-2, BOARD_W):
                            n_result = self._score_placement(
                                lookahead_board, next_piece.name,
                                next_piece.color, n_rot, n_col)
                            if n_result is not None:
                                n_score, _ = n_result
                                if n_score > best_next_score:
                                    best_next_score = n_score

                    if best_next_score > float('-inf'):
                        combined_score = base_score + 0.4 * best_next_score
                    else:
                        combined_score = base_score
                else:
                    combined_score = base_score

                if combined_score > best_score:
                    best_score = combined_score
                    best = (rot, col)
                    best_info = info
                    best_info['combined_score'] = combined_score

        return best[0], best[1], best_info


# ══════════════════════════════════════════════════════════════════════════════
#  GAME CLASS
# ══════════════════════════════════════════════════════════════════════════════
class TetrisGame:
    def __init__(self, mode="human"):
        self.mode       = mode
        self.ai_mode    = mode in ("watch", "custom")
        self.custom_mode = mode == "custom"
        self.board      = Board()
        self.piece      = Piece()
        self.next_piece = Piece()
        self.score      = 0
        self.level      = 1
        self.lines      = 0
        self.paused     = False
        self.game_over  = False
        self.result     = None   # "won" | "lost"
        self.win_lines_target = 40

        # Heuristic tuning state
        self.weights = dict(DEFAULT_WEIGHTS)
        self.weight_sliders = self._build_weight_sliders() if self.custom_mode else []

        # Timing
        self.fall_delay   = 0.5    # seconds between auto‑drops
        self.last_fall    = time.time()
        self.lock_timer   = None   # lock delay timer (human mode)
        self.lock_delay   = 0.4    # seconds before auto-lock

        # AI state
        self.ai         = AISolver(self.weights)
        self.ai_target_rot = 0
        self.ai_target_x   = 0
        self.ai_info       = {}
        self.ai_move_timer = 0
        self.ai_move_delay = 0.05     # seconds between each AI micro-step
        self.ai_thinking   = False
        self.ai_history    = []       # last N evaluations for graph
        self.ai_rot_attempts = 0      # track rotation attempts to prevent stuck
        if self.custom_mode:
            self._layout_weight_sliders()
        self._plan_ai_move()

    def _end_game(self, result):
        self.game_over = True
        self.result = result
        self.paused = False
        self.ai_thinking = False

    def _build_weight_sliders(self):
        return [
            WeightSlider('complete_lines', 'Lines +', (100, 255, 100), self.weights['complete_lines']),
            WeightSlider('aggregate_height', 'Height -', (255, 120, 120), self.weights['aggregate_height']),
            WeightSlider('holes', 'Holes -', (255, 180, 80), self.weights['holes']),
            WeightSlider('bumpiness', 'Bumpiness -', (180, 120, 255), self.weights['bumpiness']),
            WeightSlider('max_height', 'Max H -', (255, 80, 80), self.weights['max_height']),
            WeightSlider('well_depth', 'Wells -', (100, 200, 255), self.weights['well_depth']),
        ]

    def _layout_weight_sliders(self):
        if not self.weight_sliders:
            return
        px = MARGIN + BOARD_W * CELL + MARGIN
        x = px + 16
        width = PANEL_W - 32
        y = MARGIN + 338
        for slider in self.weight_sliders:
            slider.layout(x, y, width)
            y += 30

    def _sync_weight_sliders(self):
        for slider in self.weight_sliders:
            slider.set_value(self.weights[slider.key])

    def _set_weight(self, key, value):
        if key not in self.weights:
            return
        new_value = clamp(value, CUSTOM_WEIGHT_RANGE[0], CUSTOM_WEIGHT_RANGE[1])
        if abs(self.weights[key] - new_value) < 1e-6:
            return
        self.weights[key] = new_value
        # Also update the AI solver's weights
        self.ai.weights[key] = new_value
        self._sync_weight_sliders()
        if self.ai_mode and not self.game_over:
            self._plan_ai_move()

    def reset_weights(self):
        for key, value in DEFAULT_WEIGHTS.items():
            self.weights[key] = value
        # Also update the AI solver's weights
        for key, value in DEFAULT_WEIGHTS.items():
            self.ai.weights[key] = value
        self._sync_weight_sliders()
        if self.ai_mode and not self.game_over:
            self._plan_ai_move()

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _plan_ai_move(self):
        if not self.ai_mode:
            return
        rot, x, info = self.ai.find_best_move(
            self.board, self.piece, self.next_piece)
        self.ai_target_rot = rot
        self.ai_target_x   = x
        self.ai_info       = info
        self.ai_thinking   = True
        self.ai_move_timer = time.time()
        self.ai_rot_attempts = 0

    def _ghost_cells(self):
        ghost = Piece(self.piece.name)
        ghost.rotation = self.piece.rotation
        ghost.x = self.piece.x
        ghost.y = self.piece.y
        while self.board.is_valid(ghost.moved(dy=1)):
            ghost.y += 1
        return ghost.cells

    def _lock_piece(self):
        if any(y < 0 for (x, y) in self.piece.cells):
            self._end_game("lost")
            return

        self.board.lock(self.piece.cells, self.piece.color)
        cleared = self.board.clear_lines()
        self.lines += cleared
        self.score += [0, 100, 300, 500, 800][cleared] * self.level
        self.level = self.lines // 10 + 1
        self.fall_delay = max(0.05, 0.5 - (self.level - 1) * 0.04)

        if self.lines >= self.win_lines_target:
            self._end_game("won")
            return

        self.piece = self.next_piece
        self.next_piece = Piece()
        self.last_fall = time.time()
        self.lock_timer = None  # reset lock timer

        # Check game over — if the new piece overlaps existing blocks
        # even above the visible board, it's game over
        for (x, y) in self.piece.cells:
            if y >= 0 and self.board.grid[y][x] is not None:
                self._end_game("lost")
                return

        if self.ai_mode:
            self._plan_ai_move()
            if self.ai_info:
                self.ai_history.append(self.ai_info.copy())
                if len(self.ai_history) > 50:
                    self.ai_history.pop(0)

    # ── Human Input ──────────────────────────────────────────────────────────
    def handle_event(self, event):
        if self.game_over:
            return

        if self.custom_mode:
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP):
                changed = False
                for slider in self.weight_sliders:
                    if slider.handle_event(event):
                        self._set_weight(slider.key, slider.value)
                        changed = True
                if changed:
                    self._layout_weight_sliders()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.reset_weights()
            return

        if self.ai_mode:
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                if self.board.is_valid(self.piece.moved(dx=-1)):
                    self.piece.x -= 1
                    # Reset lock timer on successful move
                    if self.lock_timer is not None:
                        self.lock_timer = time.time()
            elif event.key == pygame.K_RIGHT:
                if self.board.is_valid(self.piece.moved(dx=1)):
                    self.piece.x += 1
                    if self.lock_timer is not None:
                        self.lock_timer = time.time()
            elif event.key == pygame.K_DOWN:
                if self.board.is_valid(self.piece.moved(dy=1)):
                    self.piece.y += 1
                    self.score += 1
                    self.lock_timer = None  # moved down, not on ground anymore
            elif event.key == pygame.K_UP:
                new_rot = (self.piece.rotation + 1) % 4
                new_cells = self.piece.rotated(new_rot)
                if self.board.is_valid(new_cells):
                    self.piece.rotation = new_rot
                    if self.lock_timer is not None:
                        self.lock_timer = time.time()
                else:
                    # Wall kick
                    for dx in [1, -1, 2, -2]:
                        kicked = [(x+dx, y) for x, y in new_cells]
                        if self.board.is_valid(kicked):
                            self.piece.rotation = new_rot
                            self.piece.x += dx
                            if self.lock_timer is not None:
                                self.lock_timer = time.time()
                            break
            elif event.key == pygame.K_SPACE:
                while self.board.is_valid(self.piece.moved(dy=1)):
                    self.piece.y += 1
                    self.score += 2
                self._lock_piece()
            elif event.key == pygame.K_p:
                self.paused = not self.paused

    # ── Update ───────────────────────────────────────────────────────────────
    def update(self):
        if self.paused or self.game_over:
            return

        now = time.time()

        if self.ai_mode and self.ai_thinking:
            if now - self.ai_move_timer >= self.ai_move_delay:
                self.ai_move_timer = now

                # Rotate first (with stuck detection)
                if self.piece.rotation != self.ai_target_rot:
                    # Try shortest rotation path
                    cur = self.piece.rotation
                    target = self.ai_target_rot
                    # Determine direction: clockwise or counter-clockwise
                    cw_steps = (target - cur) % 4
                    ccw_steps = (cur - target) % 4
                    if cw_steps <= ccw_steps:
                        new_rot = (cur + 1) % 4
                    else:
                        new_rot = (cur - 1) % 4

                    new_cells = self.piece.rotated(new_rot)
                    if self.board.is_valid(new_cells):
                        self.piece.rotation = new_rot
                        self.ai_rot_attempts = 0
                    else:
                        # Try wall kicks for rotation
                        kicked = False
                        for dx in [1, -1, 2, -2]:
                            kick_cells = [(x+dx, y) for x, y in new_cells]
                            if self.board.is_valid(kick_cells):
                                self.piece.rotation = new_rot
                                self.piece.x += dx
                                self.ai_rot_attempts = 0
                                kicked = True
                                break
                        if not kicked:
                            self.ai_rot_attempts += 1
                            # Give up on rotation after 4 failed attempts
                            if self.ai_rot_attempts >= 4:
                                self.ai_target_rot = self.piece.rotation
                                self.ai_rot_attempts = 0

                # Then slide horizontally
                elif self.piece.x < self.ai_target_x:
                    if self.board.is_valid(self.piece.moved(dx=1)):
                        self.piece.x += 1
                    else:
                        # Can't reach target, accept current position
                        self.ai_target_x = self.piece.x
                elif self.piece.x > self.ai_target_x:
                    if self.board.is_valid(self.piece.moved(dx=-1)):
                        self.piece.x -= 1
                    else:
                        self.ai_target_x = self.piece.x
                else:
                    # At target position — hard drop
                    while self.board.is_valid(self.piece.moved(dy=1)):
                        self.piece.y += 1
                    self._lock_piece()
                    # _lock_piece() already calls _plan_ai_move() which
                    # sets ai_thinking = True for the next piece.
                    # Do NOT reset ai_thinking here.

        # Gravity for human mode (with lock delay)
        if not self.ai_mode:
            if now - self.last_fall >= self.fall_delay:
                self.last_fall = now
                if self.board.is_valid(self.piece.moved(dy=1)):
                    self.piece.y += 1
                    self.lock_timer = None  # still falling
                else:
                    # Piece can't move down — start or check lock timer
                    if self.lock_timer is None:
                        self.lock_timer = now
                    elif now - self.lock_timer >= self.lock_delay:
                        self._lock_piece()

        # Gravity for AI (while not thinking / after position reached)
        if self.ai_mode and not self.ai_thinking:
            if now - self.last_fall >= self.fall_delay:
                self.last_fall = now
                if self.board.is_valid(self.piece.moved(dy=1)):
                    self.piece.y += 1
                else:
                    self._lock_piece()


# ══════════════════════════════════════════════════════════════════════════════
#  RENDERER
# ══════════════════════════════════════════════════════════════════════════════
class Renderer:
    def __init__(self, screen):
        self.screen = screen
        pygame.font.init()
        self.font_big   = pygame.font.SysFont("Segoe UI", 28, bold=True)
        self.font_med   = pygame.font.SysFont("Segoe UI", 20, bold=True)
        self.font_sm    = pygame.font.SysFont("Segoe UI", 15)
        self.font_xs    = pygame.font.SysFont("Segoe UI", 13)
        self.font_tag   = pygame.font.SysFont("Segoe UI", 12, bold=True)
        self.board_ox   = MARGIN
        self.board_oy   = MARGIN
        self.panel_x    = MARGIN + BOARD_W * CELL + MARGIN
        self.panel_y    = MARGIN
        self.panel_h    = BOARD_H * CELL

    # ── Utilities ─────────────────────────────────────────────────────────────
    def _cell_rect(self, gx, gy):
        return pygame.Rect(self.board_ox + gx*CELL,
                           self.board_oy + gy*CELL,
                           CELL, CELL)

    def draw_cell(self, gx, gy, color, alpha=255):
        r = self._cell_rect(gx, gy)
        surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        c = (*color, alpha)
        pygame.draw.rect(surf, c, (0, 0, CELL, CELL), border_radius=4)
        # Highlight edge
        h = (min(255, color[0]+80), min(255, color[1]+80), min(255, color[2]+80), alpha)
        pygame.draw.rect(surf, h, (0, 0, CELL, CELL), 2, border_radius=4)
        self.screen.blit(surf, r.topleft)

    def _text(self, txt, font, color, x, y, align='left'):
        surf = font.render(txt, True, color)
        if align == 'center':
            x -= surf.get_width() // 2
        elif align == 'right':
            x -= surf.get_width()
        self.screen.blit(surf, (x, y))
        return surf.get_height()

    def _draw_background(self):
        for y in range(WINDOW_H):
            ratio = y / max(WINDOW_H - 1, 1)
            r = int(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * ratio)
            g = int(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * ratio)
            b = int(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (WINDOW_W, y))

        glow = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        pygame.draw.circle(glow, (0, 200, 255, 22), (WINDOW_W - 90, 90), 180)
        pygame.draw.circle(glow, (255, 182, 72, 16), (120, WINDOW_H - 120), 220)
        pygame.draw.circle(glow, (120, 90, 255, 14), (WINDOW_W // 2, 120), 260)
        self.screen.blit(glow, (0, 0))

    def _panel_shell(self, rect, accent=PANEL_EDGE):
        shadow = pygame.Surface((rect.width + 18, rect.height + 18), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 90), shadow.get_rect(), border_radius=20)
        self.screen.blit(shadow, (rect.x - 5, rect.y + 9))
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=20)
        pygame.draw.rect(self.screen, accent, rect, 2, border_radius=20)

    def _card(self, rect, fill=DARK_GRAY, accent=None):
        pygame.draw.rect(self.screen, fill, rect, border_radius=14)
        if accent is not None:
            pygame.draw.rect(self.screen, accent, rect, 2, border_radius=14)

    def _badge(self, text, x, y, color, fill=None):
        fill = fill or (18, 22, 38)
        pad_x, pad_y = 12, 5
        surf = self.font_tag.render(text, True, color)
        rect = pygame.Rect(x, y, surf.get_width() + pad_x * 2, surf.get_height() + pad_y * 2)
        pygame.draw.rect(self.screen, fill, rect, border_radius=999)
        pygame.draw.rect(self.screen, color, rect, 1, border_radius=999)
        self.screen.blit(surf, (rect.x + pad_x, rect.y + pad_y))
        return rect

    def _stat_card(self, rect, label, value, color):
        self._card(rect, fill=(20, 26, 46), accent=color)
        self._text(label, self.font_xs, GRAY, rect.centerx, rect.y + 8, 'center')
        self._text(str(value), self.font_big, WHITE, rect.centerx, rect.y + 24, 'center')

    # ── Main draw ─────────────────────────────────────────────────────────────
    def draw(self, game: TetrisGame):
        self._draw_background()
        self._draw_board_bg()
        self._draw_locked(game.board)
        if not game.game_over and not game.paused:
            self._draw_ghost(game)
            self._draw_piece(game.piece)
        self._draw_border()

        if game.ai_mode:
            if game.custom_mode:
                self._draw_custom_ai_panel(game)
            else:
                self._draw_ai_panel(game)
        else:
            self._draw_human_panel(game)

        if game.paused:
            self._overlay("PAUSED", "(Press P to Resume)")
        if game.game_over:
            if game.result == "won":
                title = "U WON"
            else:
                title = "U LOST"
            self._overlay(title, "Press ENTER to return to the start screen")

        pygame.display.flip()

    def _draw_board_bg(self):
        board_rect = pygame.Rect(self.board_ox - 6, self.board_oy - 6,
                                 BOARD_W * CELL + 12, BOARD_H * CELL + 12)
        pygame.draw.rect(self.screen, (9, 12, 24), board_rect, border_radius=18)
        pygame.draw.rect(self.screen, (40, 48, 74), board_rect, 1, border_radius=18)
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                r = self._cell_rect(x, y)
                pygame.draw.rect(self.screen, GRID_COLOR, r, border_radius=4)

    def _draw_locked(self, board):
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                c = board.grid[y][x]
                if c:
                    self.draw_cell(x, y, c)

    def _draw_piece(self, piece):

        for (x, y) in piece.cells:
            if y >= 0:
                self.draw_cell(x, y, piece.color)

    def _draw_ghost(self, game):
        for (x, y) in game._ghost_cells():
            if y >= 0:
                self.draw_cell(x, y, game.piece.color, alpha=55)

    def _draw_border(self):
        pygame.draw.rect(self.screen, PANEL_EDGE,
                         (self.board_ox-2, self.board_oy-2,
                          BOARD_W*CELL+4, BOARD_H*CELL+4), 2, border_radius=4)

    # ── Human panel ───────────────────────────────────────────────────────────
    def _draw_human_panel(self, game):
        px, py = self.panel_x, self.panel_y
        panel_rect = pygame.Rect(px, py, PANEL_W, self.panel_h)
        self._panel_shell(panel_rect, accent=(95, 150, 255))

        y = py + 16
        self._text("TETRIS", self.font_big, WHITE, px + PANEL_W // 2, y, 'center')
        y += 32
        self._badge("HUMAN MODE", px + PANEL_W // 2 - 58, y, ACCENT, fill=(16, 28, 38))
        y += 26

        card_w = (PANEL_W - 44) // 3
        card_h = 70
        stats = [("SCORE", game.score, ACCENT), ("LEVEL", game.level, ACCENT_2), ("LINES", game.lines, SUCCESS)]
        for i, (label, val, color) in enumerate(stats):
            r = pygame.Rect(px + 16 + i * (card_w + 6), y, card_w, card_h)
            self._stat_card(r, label, val, color)

        y += card_h + 14

        next_card = pygame.Rect(px + 16, y, PANEL_W - 32, 122)
        self._card(next_card, fill=(18, 24, 42), accent=ACCENT_2)
        self._text("NEXT PIECE", self.font_xs, GRAY, next_card.x + 12, next_card.y + 10)
        self._draw_next_piece(game.next_piece, next_card.x + 12, next_card.y + 28)

        y += 138

        controls_card = pygame.Rect(px + 16, y, PANEL_W - 32, 138)
        self._card(controls_card, fill=(18, 24, 42), accent=PANEL_EDGE)
        self._text("CONTROLS", self.font_xs, GRAY, controls_card.x + 12, controls_card.y + 10)
        controls = ["← → Move", "↑ Rotate", "↓ Soft Drop",
                    "SPACE Hard Drop", "P Pause", "ESC Menu"]
        for i, line in enumerate(controls):
            self._text(line, self.font_xs, WHITE if i < 3 else GRAY,
                       controls_card.x + 14, controls_card.y + 32 + i * 15)

    # ── AI panel ────────────────────────────────────────────────────────────────
    def _draw_ai_panel(self, game):
        px, py = self.panel_x, self.panel_y
        panel_rect = pygame.Rect(px, py, PANEL_W, self.panel_h)
        self._panel_shell(panel_rect, accent=(0, 200, 255))

        y = py + 14

        # Title
        y += self._text("AI HEURISTIC ENGINE", self.font_med,
                         ACCENT, px + PANEL_W//2, y, 'center') + 4

        self._badge("WATCH MODE", px + PANEL_W // 2 - 54, y, SUCCESS, fill=(16, 28, 38))
        y += 28
        pygame.draw.line(self.screen, PANEL_EDGE,
                         (px+10, y), (px+PANEL_W-10, y))
        y += 10

        # Score / level / lines
        card_w = (PANEL_W - 44) // 3
        card_h = 68
        stats = [("SCORE", game.score, ACCENT), ("LEVEL", game.level, ACCENT_2), ("LINES", game.lines, SUCCESS)]
        for i, (label, val, color) in enumerate(stats):
            r = pygame.Rect(px + 16 + i * (card_w + 6), y, card_w, card_h)
            self._stat_card(r, label, val, color)

        y += card_h + 14

        info_card = pygame.Rect(px + 16, y, PANEL_W - 32, 110)
        self._card(info_card, fill=(18, 24, 42), accent=(125, 95, 255))
        self._text("CURRENT BOARD", self.font_xs, GRAY, info_card.x + 12, info_card.y + 10)

        self._text(f"Lines cleared: {game.lines}", self.font_xs, WHITE, info_card.x + 12, info_card.y + 34)
        self._text(f"Max height: {game.board.max_height()}", self.font_xs, WHITE, info_card.x + 12, info_card.y + 49)
        self._text(f"Holes: {game.board.count_holes()}", self.font_xs, WHITE, info_card.x + 12, info_card.y + 64)
        self._text(f"Bumpiness: {game.board.bumpiness()}", self.font_xs, WHITE, info_card.x + 12, info_card.y + 79)

        y += 122

        pygame.draw.line(self.screen, PANEL_EDGE, (px+10, y), (px+PANEL_W-10, y))
        y += 8

        # ── Current board state ─────────────────────────────────────────────
        y += self._text("LIVE HEURISTICS", self.font_xs, GRAY, px+16, y) + 8

        board = game.board
        heights  = board.column_heights()
        lines    = game.lines
        agg_h    = sum(heights)
        holes    = board.count_holes()
        bump     = board.bumpiness()
        max_h    = board.max_height()
        wells    = board.well_depth()

        # All 6 heuristic metrics with colors and max values for scaling
        heuristics = [
            ("Lines Cleared", lines,   (100, 255, 100), 50),
            ("Agg. Height",   agg_h,   (255, 120, 120), 200),
            ("Max Height",    max_h,   (255,  80,  80), 20),
            ("Holes",         holes,   (255, 180,  80), 30),
            ("Bumpiness",     bump,    (180, 120, 255), 80),
            ("Well Depth",    wells,   (100, 200, 255), 30),
        ]
        
        for label, val, color, max_val in heuristics:
            # Draw label and value side by side
            label_text = f"{label}"
            val_text = f"{val}"
            label_surf = self.font_xs.render(label_text, True, GRAY)
            val_surf = self.font_xs.render(val_text, True, color)
            
            self.screen.blit(label_surf, (px+16, y))
            self.screen.blit(val_surf, (px + PANEL_W - 16 - val_surf.get_width(), y))
            y += 18
            
            # Draw progress bar
            bar_h = 6
            bar_w_max = PANEL_W - 36
            bar_w = max(0, min(bar_w_max, int((val / max(max_val, 1)) * bar_w_max)))
            pygame.draw.rect(self.screen, DARK_GRAY, (px+16, y, bar_w_max, bar_h), border_radius=3)
            pygame.draw.rect(self.screen, color, (px+16, y, bar_w, bar_h), border_radius=3)
            y += 12

        y += 4
        pygame.draw.line(self.screen, PANEL_EDGE, (px+10, y), (px+PANEL_W-10, y))
        y += 8

        # ── Best move info ────────────────────────────────────────────────────
        y += self._text("BEST MOVE", self.font_xs, GRAY, px+16, y) + 6

        info = game.ai_info
        if info:
            rot_names = ["0°", "90°", "180°", "270°"]

            # Rotation and column on same row
            rot_text = f"Rotation: {rot_names[info.get('rotation',0)]}"
            col_text = f"Col: {info.get('column', '?')}"
            rot_surf = self.font_xs.render(rot_text, True, (160,220,255))
            col_surf = self.font_xs.render(col_text, True, (160,220,255))
            
            self.screen.blit(rot_surf, (px+16, y))
            self.screen.blit(col_surf, (px + PANEL_W - 16 - col_surf.get_width(), y))
            y += 18
            
            score_text = f"Score: {info.get('score', 0):.4f}"
            y += self._text(score_text, self.font_xs, (100,255,180), px+16, y) + 4

        # Next piece
        self._draw_next_piece(game.next_piece, px + PANEL_W//2 - 32, py + 14)

    def _draw_custom_ai_panel(self, game):
        px, py = self.panel_x, self.panel_y
        panel_rect = pygame.Rect(px, py, PANEL_W, self.panel_h)
        self._panel_shell(panel_rect, accent=(255, 182, 72))

        y = py + 14
        y += self._text("CUSTOM HEURISTIC", self.font_med,
                         ACCENT_2, px + PANEL_W // 2, y, 'center') + 4
        self._badge("LIVE TUNING", px + PANEL_W // 2 - 54, y, SUCCESS, fill=(16, 28, 38))
        y += 28
        pygame.draw.line(self.screen, PANEL_EDGE,
                         (px+10, y), (px+PANEL_W-10, y))
        y += 10

        card_w = (PANEL_W - 44) // 3
        card_h = 68
        stats = [("SCORE", game.score, ACCENT), ("LEVEL", game.level, ACCENT_2), ("LINES", game.lines, SUCCESS)]
        for i, (label, val, color) in enumerate(stats):
            r = pygame.Rect(px + 16 + i * (card_w + 6), y, card_w, card_h)
            self._stat_card(r, label, val, color)

        y += card_h + 14

        info_card = pygame.Rect(px + 16, y, PANEL_W - 32, 96)
        self._card(info_card, fill=(18, 24, 42), accent=(255, 182, 72))
        self._text("CURRENT BOARD", self.font_xs, GRAY, info_card.x + 12, info_card.y + 10)
        self._text(f"Max height: {game.board.max_height()}", self.font_xs, WHITE, info_card.x + 12, info_card.y + 34)
        self._text(f"Holes: {game.board.count_holes()}", self.font_xs, WHITE, info_card.x + 12, info_card.y + 49)
        self._text(f"Bumpiness: {game.board.bumpiness()}", self.font_xs, WHITE, info_card.x + 12, info_card.y + 64)

        y += 108
        pygame.draw.line(self.screen, PANEL_EDGE, (px+10, y), (px+PANEL_W-10, y))
        y += 8

        y += self._text("LIVE WEIGHTS", self.font_xs, GRAY, px+16, y) + 6
        game._layout_weight_sliders()
        for slider in game.weight_sliders:
            slider.draw(self.screen, self.font_xs, self.font_tag)

        if game.weight_sliders:
            y = game.weight_sliders[-1].row_rect.bottom + 8

        pygame.draw.line(self.screen, PANEL_EDGE, (px+10, y), (px+PANEL_W-10, y))
        y += 8

        info = game.ai_info
        if info:
            rot_names = ["0°", "90°", "180°", "270°"]
            y += self._text("CURRENT BEST MOVE", self.font_xs, GRAY, px+16, y) + 4
            y += self._text(f"Rotation : {rot_names[info.get('rotation',0)]}",
                             self.font_xs, (160,220,255), px+16, y) + 2
            y += self._text(f"Column   : {info.get('column', '?')}",
                             self.font_xs, (160,220,255), px+16, y) + 2
            y += self._text(f"Score    : {info.get('score', 0):.4f}",
                             self.font_xs, (100,255,180), px+16, y) + 4

        self._text("Drag sliders to update the AI live. Press R to reset.",
                   self.font_xs, GRAY, px + PANEL_W // 2, self.panel_h - 22, 'center')

    def _draw_next_piece(self, piece, ox, oy):
        mini = 20
        for (dx, dy) in SHAPES[piece.name][0]:
            r = pygame.Rect(ox + dx*mini, oy + dy*mini, mini-1, mini-1)
            pygame.draw.rect(self.screen, piece.color, r, border_radius=3)
            pygame.draw.rect(self.screen, tuple(min(255,c+60) for c in piece.color), r, 1, border_radius=3)

    def _overlay(self, title, sub):
        surf = pygame.Surface((BOARD_W*CELL, BOARD_H*CELL), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 170))
        self.screen.blit(surf, (self.board_ox, self.board_oy))
        cx = self.board_ox + BOARD_W*CELL//2
        cy = self.board_oy + BOARD_H*CELL//2
        self._text(title, self.font_big, WHITE, cx, cy-20, 'center')
        self._text(sub,   self.font_sm,  GRAY,  cx, cy+14, 'center')


# ══════════════════════════════════════════════════════════════════════════════
#  MENU SCREEN
# ══════════════════════════════════════════════════════════════════════════════
class MenuScreen:
    def __init__(self, screen):
        self.screen = screen
        self.font_title  = pygame.font.SysFont("Segoe UI", 60, bold=True)
        self.font_sub    = pygame.font.SysFont("Segoe UI", 22)
        self.font_btn    = pygame.font.SysFont("Segoe UI", 26, bold=True)
        self.font_desc   = pygame.font.SysFont("Segoe UI", 15)
        self.selected    = 0
        self.t           = 0.0
        self.options     = [
            ("🎮  Play as Human",   "Classic keyboard-controlled Tetris",              "human"),
            ("🤖  Watch AI Play",   "6-feature heuristic AI with lookahead",           "watch"),
            ("🛠  Make Your Own Heuristic", "Live sliders for all 6 weights",       "custom"),
        ]

    def _badge(self, text, x, y, color, fill=None):
        fill = fill or (18, 22, 38)
        pad_x, pad_y = 12, 5
        surf = self.font_desc.render(text, True, color)
        rect = pygame.Rect(x, y, surf.get_width() + pad_x * 2, surf.get_height() + pad_y * 2)
        pygame.draw.rect(self.screen, fill, rect, border_radius=999)
        pygame.draw.rect(self.screen, color, rect, 1, border_radius=999)
        self.screen.blit(surf, (rect.x + pad_x, rect.y + pad_y))
        return rect

    def draw(self):
        self.t += 0.02
        # Animated gradient background
        for y in range(WINDOW_H):
            ratio = y / WINDOW_H
            r = max(0, min(255, int(8 + 16 * math.sin(self.t + ratio * 3.2))))
            g = max(0, min(255, int(12 + 10 * math.sin(self.t * 0.7 + ratio * 2.2))))
            b = max(0, min(255, int(24 + 28 * math.sin(self.t * 0.5 + ratio))))
            pygame.draw.line(self.screen, (r, g, b), (0, y), (WINDOW_W, y))

        mist = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        pygame.draw.circle(mist, (0, 200, 255, 28), (WINDOW_W - 120, 100), 190)

        pygame.draw.circle(mist, (255, 182, 72, 18), (120, WINDOW_H - 120), 220)
        pygame.draw.circle(mist, (255, 255, 255, 10), (WINDOW_W // 2, 160), 300)
        self.screen.blit(mist, (0, 0))

        cx = WINDOW_W // 2

        card = pygame.Rect(cx - 250, 58, 500, 470)
        shadow = pygame.Surface((card.width + 18, card.height + 18), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 92), shadow.get_rect(), border_radius=28)
        self.screen.blit(shadow, (card.x - 6, card.y + 10))

        pygame.draw.rect(self.screen, (14, 18, 34), card, border_radius=28)
        pygame.draw.rect(self.screen, ACCENT, card, 2, border_radius=28)

        # Animated title
        wave_y = int(4 * math.sin(self.t * 2))
        title = self.font_title.render("TETRIS", True, WHITE)
        glow_colors = [(0, 80, 120), (0, 120, 180), (0, 180, 220), (0, 220, 255), (255, 255, 255)]
        for i, gc in enumerate(glow_colors):
            ts = self.font_title.render("TETRIS", True, gc)
            self.screen.blit(ts, (cx - ts.get_width() // 2 + (i - 2) * 0.5,
                                  86 + wave_y + (i - 2) * 0.5))

        sub = self.font_sub.render("AI EDITION", True, ACCENT)
        self.screen.blit(sub, (cx - sub.get_width() // 2, 156))

        # Draw two badges centered under the subtitle
        b1_text = "Win: clear 40 lines"
        b2_text = "Lose: stack reaches the roof"
        pad_x, pad_y = 12, 5
        b1_surf = self.font_desc.render(b1_text, True, SUCCESS)
        b2_surf = self.font_desc.render(b2_text, True, DANGER)
        b1_w = b1_surf.get_width() + pad_x * 2
        b2_w = b2_surf.get_width() + pad_x * 2
        spacing = 14
        total_w = b1_w + spacing + b2_w
        start_x = cx - total_w // 2
        b_y = 194
        # badge 1
        rect1 = pygame.Rect(start_x, b_y, b1_w, b1_surf.get_height() + pad_y * 2)
        pygame.draw.rect(self.screen, (16, 28, 38), rect1, border_radius=999)
        pygame.draw.rect(self.screen, SUCCESS, rect1, 1, border_radius=999)
        self.screen.blit(b1_surf, (rect1.x + pad_x, rect1.y + pad_y))
        # badge 2
        rect2 = pygame.Rect(start_x + b1_w + spacing, b_y, b2_w, b2_surf.get_height() + pad_y * 2)
        pygame.draw.rect(self.screen, (28, 18, 24), rect2, border_radius=999)
        pygame.draw.rect(self.screen, DANGER, rect2, 1, border_radius=999)
        self.screen.blit(b2_surf, (rect2.x + pad_x, rect2.y + pad_y))

        # Buttons
        btn_w, btn_h = 400, 72
        for i, (label, desc, _) in enumerate(self.options):
            bx = cx - btn_w//2
            by = 245 + i * 100
            # Glow on hover
            hover = (self.selected == i)
            if hover:
                for expand in [12, 8, 4]:
                    s = pygame.Surface((btn_w+expand*2, btn_h+expand*2), pygame.SRCALPHA)
                    alpha = max(0, 60 - expand * 5)
                    color = (0, 160, 220, alpha)
                    pygame.draw.rect(s, color, (0,0,btn_w+expand*2, btn_h+expand*2), border_radius=18)
                    self.screen.blit(s, (bx-expand, by-expand))

            bg_col = (20, 80, 160) if hover else (25, 25, 50)
            border = (0, 200, 255) if hover else (60, 60, 100)
            pygame.draw.rect(self.screen, bg_col, (bx, by, btn_w, btn_h), border_radius=14)
            pygame.draw.rect(self.screen, border, (bx, by, btn_w, btn_h), 2, border_radius=14)

            # removed thin white overlay used previously for hover highlight

            ts = self.font_btn.render(label, True, WHITE)
            self.screen.blit(ts, (bx + 24, by + 12))
            ds = self.font_desc.render(desc, True, GRAY)
            self.screen.blit(ds, (bx + 26, by + 42))

        # Navigation hint
        hint = self.font_desc.render("↑ ↓ Navigate    ENTER Select", True, (90, 102, 132))
        self.screen.blit(hint, (cx - hint.get_width()//2, WINDOW_H - 44))

        win_text = self.font_desc.render("Win condition: clear 40 lines", True, ACCENT)
        self.screen.blit(win_text, (cx - win_text.get_width()//2, WINDOW_H - 66))

        pygame.display.flip()

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_UP, pygame.K_w):
                self.selected = (self.selected - 1) % len(self.options)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.selected = (self.selected + 1) % len(self.options)
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                return self.options[self.selected][2]
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Tetris AI Edition")
    
    clock  = pygame.time.Clock()

    state  = "menu"      # "menu" | "game"
    menu   = MenuScreen(screen)
    game   = None
    renderer = None

    while True:
        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if state == "menu":
                result = menu.handle_event(event)
                if result is not None:
                    game     = TetrisGame(mode=result)
                    renderer = Renderer(screen)
                    state    = "game"
            elif state == "game":
                if game and game.game_over and event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_ESCAPE):
                    state = "menu"
                    menu  = MenuScreen(screen)
                    game  = None
                    renderer = None
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    state = "menu"
                    menu  = MenuScreen(screen)
                    game  = None
                    renderer = None
                else:
                    game.handle_event(event)

        # ── Update ─────────────────────────────────────────────────────────────
        if state == "game" and game:
            game.update()

        # ── Draw ───────────────────────────────────────────────────────────────
        if state == "menu":
            menu.draw()
        elif state == "game" and game and renderer:
            renderer.draw(game)

        clock.tick(FPS)

  
if __name__ == "__main__":
    main() 

