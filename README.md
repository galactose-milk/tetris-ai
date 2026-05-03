# 🎮 Tetris AI Edition

> A Tetris game in Python and Pygame with a heuristic AI that evaluates every possible move.

## Project Overview

This is a single-file Tetris project built for an AI course. It has two modes:

- Human mode for manual play
- AI mode where the solver chooses moves automatically

The main goal of the project is to show how a game AI can be built with search and heuristics instead of machine learning.

## AI Depth

The AI uses a deterministic heuristic evaluation function. For each piece, it tries every valid rotation and column, simulates the drop, and scores the resulting board state.

### What the AI checks

- `complete_lines`: rewards line clears
- `aggregate_height`: penalizes tall stacks
- `holes`: penalizes empty cells trapped below blocks
- `bumpiness`: penalizes uneven surfaces
- `max_height`: penalizes very high stacks
- `well_depth`: penalizes deep wells

### Decision process

1. Generate all possible placements for the current piece.
2. Simulate each placement on a copy of the board.
3. Compute the heuristic score.
4. Choose the move with the best score.
5. When the stack becomes dangerous, the solver also looks one piece ahead.

### Current weights

```python
WEIGHTS = {
    'complete_lines':    0.760666,
    'aggregate_height': -0.510066,
    'holes':            -0.35663,
    'bumpiness':        -0.184483,
    'max_height':       -0.30,
    'well_depth':       -0.15,
}
```

These weights make the AI prefer safer boards with fewer holes and better line-clearing opportunities.

## Project Structure

- `Board`: grid state, collisions, line clearing, and heuristic metrics
- `Piece`: tetromino shapes and movement
- `AISolver`: searches for the best move using the heuristic score
- `TetrisGame`: game loop, timing, scoring, and AI stepping
- `Renderer`: all drawing for the board, pieces, and panels
- `MenuScreen`: animated start menu

## How to Run

```bash
git clone https://github.com/galactose-milk/tetris-ai.git
cd tetris-ai
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pygame
python tetris_ai.py
```

## Summary for Presentation

This project demonstrates a classic AI technique: evaluating many possible actions with a hand-designed scoring function. The AI is explainable because every choice comes from measurable board features, and the right-side panel shows those values live while the game runs.