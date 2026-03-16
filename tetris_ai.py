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
BG          = (10, 10, 20)
GRID_COLOR  = (30, 30, 50)
PANEL_BG    = (18, 18, 35)
PANEL_EDGE  = (60, 60, 100)
WHITE       = (240, 240, 255)
GRAY        = (120, 120, 150)
DARK_GRAY   = (40, 40, 60)
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
# Tuned classic weights for the Tetris heuristic AI
WEIGHTS = {
    'lines_cleared':     1.0,     # reward
    'aggregate_height': -0.510066,
    'holes':            -0.35663,
    'bumpiness':        -0.184483,
}

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

    def evaluate(self):
        temp_board = self.copy()
        lines = temp_board.clear_lines()
        h     = temp_board.aggregate_height()
        holes = temp_board.count_holes()
        bump  = temp_board.bumpiness()
        score = (WEIGHTS['lines_cleared']     * lines +
                 WEIGHTS['aggregate_height']  * h     +
                 WEIGHTS['holes']             * holes  +
                 WEIGHTS['bumpiness']         * bump)
        return score, lines, h, holes, bump


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


# ══════════════════════════════════════════════════════════════════════════════
#  AI SOLVER
# ══════════════════════════════════════════════════════════════════════════════
class AISolver:
    def find_best_move(self, board: Board, piece: Piece):
        """Try all rotations × columns, return (rotation, x, heuristic_info)."""
        best_score = float('-inf')
        best       = (0, piece.x)
        best_info  = {}

        for rot in range(4):
            for col in range(-2, BOARD_W + 2):
                sim_piece = Piece(piece.name)
                sim_piece.rotation = rot
                sim_piece.x = col

                # Drop the piece down
                sim_piece.y = -2
                valid_start = False
                while True:
                    next_cells = [(sim_piece.x + dx, sim_piece.y + 1 + dy)
                                  for (dx, dy) in SHAPES[sim_piece.name][rot]]
                    if board.is_valid(next_cells):
                        sim_piece.y += 1
                        valid_start = True
                    else:
                        break

                cells = [(sim_piece.x + dx, sim_piece.y + dy)
                         for (dx, dy) in SHAPES[sim_piece.name][rot]]

                # Skip if invalid (piece never moved into board)
                if not board.is_valid(cells) and not valid_start:
                    continue
                if not all(0 <= x < BOARD_W for x, y in cells):
                    continue
                # Ensure at least one cell is inside board vertically
                if not any(0 <= y < BOARD_H for x, y in cells):
                    continue

                sim_board = board.copy()
                sim_board.lock(cells, piece.color)
                score, lines, h, holes, bump = sim_board.evaluate()

                if score > best_score:
                    best_score = score
                    best = (rot, col)
                    best_info = {
                        'score':     score,
                        'lines':     lines,
                        'height':    h,
                        'holes':     holes,
                        'bumpiness': bump,
                        'rotation':  rot,
                        'column':    col,
                    }

        return best[0], best[1], best_info


# ══════════════════════════════════════════════════════════════════════════════
#  GAME CLASS
# ══════════════════════════════════════════════════════════════════════════════
class TetrisGame:
    def __init__(self, ai_mode=False):
        self.ai_mode    = ai_mode
        self.board      = Board()
        self.piece      = Piece()
        self.next_piece = Piece()
        self.score      = 0
        self.level      = 1
        self.lines      = 0
        self.paused     = False
        self.game_over  = False

        # Timing
        self.fall_delay   = 0.5    # seconds between auto‑drops
        self.last_fall    = time.time()
        self.last_lock    = time.time()
        self.lock_delay   = 0.4

        # AI state
        self.ai         = AISolver()
        self.ai_target_rot = 0
        self.ai_target_x   = 0
        self.ai_info       = {}
        self.ai_move_timer = 0
        self.ai_move_delay = 0.07     # seconds between each AI micro-step
        self.ai_thinking   = False
        self.ai_history    = []       # last N evaluations for graph
        self._plan_ai_move()

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _plan_ai_move(self):
        if not self.ai_mode:
            return
        rot, x, info = self.ai.find_best_move(self.board, self.piece)
        self.ai_target_rot = rot
        self.ai_target_x   = x
        self.ai_info       = info
        self.ai_thinking   = True
        self.ai_move_timer = time.time()

    def _ghost_cells(self):
        ghost = Piece(self.piece.name)
        ghost.rotation = self.piece.rotation
        ghost.x = self.piece.x
        ghost.y = self.piece.y
        while self.board.is_valid(ghost.moved(dy=1)):
            ghost.y += 1
        return ghost.cells

    def _lock_piece(self):
        self.board.lock(self.piece.cells, self.piece.color)
        cleared = self.board.clear_lines()
        self.lines += cleared
        self.score += [0, 100, 300, 500, 800][cleared] * self.level
        self.level = self.lines // 10 + 1
        self.fall_delay = max(0.05, 0.5 - (self.level - 1) * 0.04)
        self.piece = self.next_piece
        self.next_piece = Piece()
        self.last_fall = time.time()

        # Check game over
        if not self.board.is_valid(self.piece.cells):
            self.game_over = True
            return

        if self.ai_mode:
            self._plan_ai_move()
            if self.ai_info:
                self.ai_history.append(self.ai_info.copy())
                if len(self.ai_history) > 50:
                    self.ai_history.pop(0)

    # ── Human Input ──────────────────────────────────────────────────────────
    def handle_event(self, event):
        if self.game_over or self.ai_mode:
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                if self.board.is_valid(self.piece.moved(dx=-1)):
                    self.piece.x -= 1
            elif event.key == pygame.K_RIGHT:
                if self.board.is_valid(self.piece.moved(dx=1)):
                    self.piece.x += 1
            elif event.key == pygame.K_DOWN:
                if self.board.is_valid(self.piece.moved(dy=1)):
                    self.piece.y += 1
                    self.score += 1
            elif event.key == pygame.K_UP:
                new_rot = (self.piece.rotation + 1) % 4
                new_cells = self.piece.rotated(new_rot)
                if self.board.is_valid(new_cells):
                    self.piece.rotation = new_rot
                else:
                    # Wall kick
                    for dx in [1, -1, 2, -2]:
                        kicked = [(x+dx, y) for x, y in new_cells]
                        if self.board.is_valid(kicked):
                            self.piece.rotation = new_rot
                            self.piece.x += dx
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
                moved = False

                # Rotate first
                if self.piece.rotation != self.ai_target_rot:
                    new_rot = (self.piece.rotation + 1) % 4
                    new_cells = self.piece.rotated(new_rot)
                    if self.board.is_valid(new_cells):
                        self.piece.rotation = new_rot
                    moved = True

                # Then slide horizontally
                elif self.piece.x < self.ai_target_x:
                    if self.board.is_valid(self.piece.moved(dx=1)):
                        self.piece.x += 1
                    moved = True
                elif self.piece.x > self.ai_target_x:
                    if self.board.is_valid(self.piece.moved(dx=-1)):
                        self.piece.x -= 1
                    moved = True
                else:
                    # Drop it fast
                    if self.board.is_valid(self.piece.moved(dy=1)):
                        self.piece.y += 1
                    else:
                        self._lock_piece()
                        self.ai_thinking = False

        # Gravity for human mode
        if not self.ai_mode:
            if now - self.last_fall >= self.fall_delay:
                self.last_fall = now
                if self.board.is_valid(self.piece.moved(dy=1)):
                    self.piece.y += 1
                else:
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

    # ── Main draw ─────────────────────────────────────────────────────────────
    def draw(self, game: TetrisGame):
        self.screen.fill(BG)
        self._draw_board_bg()
        self._draw_locked(game.board)
        if not game.game_over and not game.paused:
            self._draw_ghost(game)
            self._draw_piece(game.piece)
        self._draw_border()

        if game.ai_mode:
            self._draw_ai_panel(game)
        else:
            self._draw_human_panel(game)

        if game.paused:
            self._overlay("PAUSED", "(Press P to Resume)")
        if game.game_over:
            self._overlay("GAME OVER", f"Score: {game.score}  Press ESC")

        pygame.display.flip()

    def _draw_board_bg(self):
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                r = self._cell_rect(x, y)
                pygame.draw.rect(self.screen, GRID_COLOR, r, border_radius=2)

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
        pygame.draw.rect(self.screen, PANEL_BG,
                         (px, py, PANEL_W, self.panel_h), border_radius=8)
        pygame.draw.rect(self.screen, PANEL_EDGE,
                         (px, py, PANEL_W, self.panel_h), 2, border_radius=8)

        y = py + 16
        y += self._text("TETRIS", self.font_big, (100,200,255), px + PANEL_W//2, y, 'center') + 10

        # Score / level / lines
        for label, val in [("SCORE", game.score), ("LEVEL", game.level), ("LINES", game.lines)]:
            y += self._text(label, self.font_xs, GRAY, px+16, y) + 2
            y += self._text(str(val), self.font_big, WHITE, px+16, y) + 14

        # Next piece
        y += 10
        y += self._text("NEXT", self.font_xs, GRAY, px+16, y) + 6
        self._draw_next_piece(game.next_piece, px+16, y)
        y += 90

        # Controls
        y += 10
        y += self._text("CONTROLS", self.font_xs, GRAY, px+16, y) + 6
        for line in ["← → Move", "↑  Rotate", "↓  Soft Drop",
                     "SPACE Hard Drop", "P  Pause", "ESC  Menu"]:
            y += self._text(line, self.font_xs, (160,160,200), px+20, y) + 3

    # ── AI panel ────────────────────────────────────────────────────────────────
    def _draw_ai_panel(self, game):
        px, py = self.panel_x, self.panel_y
        pygame.draw.rect(self.screen, PANEL_BG,
                         (px, py, PANEL_W, self.panel_h), border_radius=8)
        pygame.draw.rect(self.screen, PANEL_EDGE,
                         (px, py, PANEL_W, self.panel_h), 2, border_radius=8)

        y = py + 14

        # Title
        y += self._text("AI HEURISTIC ENGINE", self.font_med,
                         (80,220,170), px + PANEL_W//2, y, 'center') + 4
        pygame.draw.line(self.screen, PANEL_EDGE,
                         (px+10, y), (px+PANEL_W-10, y))
        y += 10

        # Score / level / lines
        col2 = px + PANEL_W // 2
        y += self._text("SCORE", self.font_xs, GRAY, px+16, y) + 2
        y += self._text(str(game.score), self.font_big, WHITE, px+16, y) + 2
        self._text("LEVEL", self.font_xs, GRAY, col2, y-24)
        self._text(str(game.level), self.font_big, (255,200,50), col2, y-10)
        y += 4
        y += self._text(f"Lines cleared: {game.lines}", self.font_xs,
                         (180,180,220), px+16, y) + 10

        pygame.draw.line(self.screen, PANEL_EDGE, (px+10, y), (px+PANEL_W-10, y))
        y += 8

        # ── Current board state ─────────────────────────────────────────────
        y += self._text("CURRENT BOARD STATE", self.font_xs, GRAY, px+16, y) + 6

        board = game.board
        heights  = board.column_heights()
        agg_h    = sum(heights)
        holes    = board.count_holes()
        bump     = board.bumpiness()

        stats = [
            ("Agg. Height",  agg_h,  (255, 120, 120), 200),
            ("Holes",        holes,  (255, 180,  80), 30),
            ("Bumpiness",    bump,   (180, 120, 255), 80),
        ]
        for label, val, color, max_val in stats:
            bar_w = max(0, min(PANEL_W - 100, int((val / max(max_val, 1)) * (PANEL_W - 100))))
            y += self._text(f"{label}: {val}", self.font_xs, color, px+16, y) + 2
            pygame.draw.rect(self.screen, DARK_GRAY, (px+16, y, PANEL_W-36, 10), border_radius=4)
            pygame.draw.rect(self.screen, color, (px+16, y, bar_w, 10), border_radius=4)
            y += 14

        y += 6
        pygame.draw.line(self.screen, PANEL_EDGE, (px+10, y), (px+PANEL_W-10, y))
        y += 8

        # ── Best move info ────────────────────────────────────────────────────
        y += self._text("BEST MOVE SELECTED", self.font_xs, GRAY, px+16, y) + 6

        info = game.ai_info
        if info:
            rot_names = ["0°", "90°", "180°", "270°"]
            y += self._text(f"Rotation : {rot_names[info.get('rotation',0)]}",
                             self.font_xs, (160,220,255), px+16, y) + 3
            y += self._text(f"Column   : {info.get('column', '?')}",
                             self.font_xs, (160,220,255), px+16, y) + 3
            y += self._text(f"Score    : {info.get('score', 0):.4f}",
                             self.font_xs, (100,255,180), px+16, y) + 10

            # Component breakdown
            y += self._text("HEURISTIC BREAKDOWN", self.font_xs, GRAY, px+16, y) + 6

            components = [
                ("Lines Cleared",  info.get('lines',  0), WEIGHTS['lines_cleared'],      (100,255,100)),
                ("Aggregate Ht",   info.get('height', 0), WEIGHTS['aggregate_height'],   (255,120,120)),
                ("Holes",          info.get('holes',  0), WEIGHTS['holes'],              (255,180, 80)),
                ("Bumpiness",      info.get('bumpiness',0), WEIGHTS['bumpiness'],         (180,120,255)),
            ]

            for name, raw, weight, color in components:
                contribution = weight * raw
                y += self._text(f"{name}:", self.font_xs, WHITE, px+16, y) + 1
                y += self._text(f"  raw={raw:.1f}  w={weight:.3f}  Δ={contribution:.3f}",
                                 self.font_xs, color, px+16, y) + 5

        y += 8
        pygame.draw.line(self.screen, PANEL_EDGE, (px+10, y), (px+PANEL_W-10, y))
        y += 8

        # ── Weight legend ─────────────────────────────────────────────────────
        y += self._text("HEURISTIC WEIGHTS", self.font_xs, GRAY, px+16, y) + 6
        weight_info = [
            ("lines_cleared",     WEIGHTS['lines_cleared'],     (100,255,100)),
            ("aggregate_height",  WEIGHTS['aggregate_height'],  (255,120,120)),
            ("holes",             WEIGHTS['holes'],             (255,180, 80)),
            ("bumpiness",         WEIGHTS['bumpiness'],         (180,120,255)),
        ]
        for wname, wval, wcolor in weight_info:
            bar_fill = int(abs(wval) * 120)
            rect_x = px + 16
            pygame.draw.rect(self.screen, DARK_GRAY, (rect_x, y+2, 120, 8), border_radius=3)
            pygame.draw.rect(self.screen, wcolor, (rect_x, y+2, bar_fill, 8), border_radius=3)
            y += self._text(f"{wname}: {wval:+.4f}", self.font_xs, wcolor, px+148, y) + 11

        y += 4
        pygame.draw.line(self.screen, PANEL_EDGE, (px+10, y), (px+PANEL_W-10, y))
        y += 8

        # ── Column heights mini-graph ─────────────────────────────────────────
        remaining = self.panel_h - (y - py) - 16
        if remaining > 40:
            y += self._text("COLUMN HEIGHTS", self.font_xs, GRAY, px+16, y) + 4
            bar_area_h = min(remaining - 20, 60)
            bar_w_each = (PANEL_W - 36) // BOARD_W
            max_h = max(heights) if max(heights) > 0 else 1
            for i, h in enumerate(heights):
                bh = int((h / max_h) * bar_area_h)
                bx = px + 16 + i * bar_w_each
                by = y + bar_area_h - bh
                # Colour by height
                ratio = h / BOARD_H
                rc = (int(50 + 200*ratio), int(200 - 150*ratio), 100)
                pygame.draw.rect(self.screen, rc, (bx, by, bar_w_each-2, bh), border_radius=2)
            y += bar_area_h + 4
            self._text(f"Max={max(heights)}  Avg={agg_h//BOARD_W}",
                        self.font_xs, GRAY, px+16, y)

        # Next piece
        self._draw_next_piece(game.next_piece, px + PANEL_W//2 - 32, py + 14)

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
        self.font_title  = pygame.font.SysFont("Segoe UI", 64, bold=True)
        self.font_sub    = pygame.font.SysFont("Segoe UI", 22)
        self.font_btn    = pygame.font.SysFont("Segoe UI", 26, bold=True)
        self.font_desc   = pygame.font.SysFont("Segoe UI", 15)
        self.selected    = 0
        self.t           = 0.0
        self.options     = [
            ("🎮  Play as Human",   "Classic keyboard-controlled Tetris",              False),
            ("🤖  Watch AI Play",   "Heuristic AI with live analysis panel",            True),
        ]

    def draw(self):
        self.t += 0.02
        # Animated gradient background
        for y in range(WINDOW_H):
            ratio = y / WINDOW_H
            r = max(0, min(255, int(10 + 15*math.sin(self.t + ratio*3))))
            g = max(0, min(255, int(10 + 10*math.sin(self.t*0.7 + ratio*2))))
            b = max(0, min(255, int(20 + 30*math.sin(self.t*0.5 + ratio))))
            pygame.draw.line(self.screen, (r, g, b), (0, y), (WINDOW_W, y))

        cx = WINDOW_W // 2

        # Animated title
        wave_y = int(4 * math.sin(self.t * 2))
        title_surf = self.font_title.render("TETRIS", True, (0, 0, 0))
        # Glow layers
        glow_colors = [(0,80,120),(0,120,180),(0,180,220),(0,220,255),(255,255,255)]
        for i, gc in enumerate(glow_colors):
            ts = self.font_title.render("TETRIS", True, gc)
            self.screen.blit(ts, (cx - ts.get_width()//2 + (i-2)*0.5,
                                  80 + wave_y + (i-2)*0.5))

        sub = self.font_sub.render("A I   E D I T I O N", True, (100,200,255))
        self.screen.blit(sub, (cx - sub.get_width()//2, 155))

        # Buttons
        btn_w, btn_h = 400, 80
        for i, (label, desc, _) in enumerate(self.options):
            bx = cx - btn_w//2
            by = 240 + i * 110
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

            ts = self.font_btn.render(label, True, WHITE)
            self.screen.blit(ts, (bx + 24, by + 14))
            ds = self.font_desc.render(desc, True, GRAY)
            self.screen.blit(ds, (bx + 26, by + 48))

        # Navigation hint
        hint = self.font_desc.render("↑ ↓  Navigate    ENTER  Select", True, (80,80,110))
        self.screen.blit(hint, (cx - hint.get_width()//2, WINDOW_H - 40))

        pygame.display.flip()

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_UP, pygame.K_w):
                self.selected = (self.selected - 1) % len(self.options)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.selected = (self.selected + 1) % len(self.options)
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                return self.options[self.selected][2]  # returns ai_mode bool
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
                    game     = TetrisGame(ai_mode=result)
                    renderer = Renderer(screen)
                    state    = "game"
            elif state == "game":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    state = "menu"
                    menu  = MenuScreen(screen)
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
