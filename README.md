# 🎮 Tetris AI Edition

> A feature-complete Tetris game built with Python & Pygame — play it yourself or sit back and watch a heuristic AI crush it in real time!

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-2.x-green?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎮 **Human Mode** | Full keyboard-controlled Tetris with ghost piece, pause, and hard-drop |
| 🤖 **AI Mode** | Heuristic-driven AI that evaluates every possible placement for each piece |
| 📊 **Live Heuristic Panel** | Real-time sidebar showing the AI's decision-making — weights, scores, breakdowns |
| 📈 **Column Height Graph** | Visual mini-graph of all 10 column heights updated every frame |
| 🌊 **Animated Menu** | Wave-animated gradient background with glowing title and hover effects |
| 👻 **Ghost Piece** | Translucent preview showing where the current piece will land |
| ⚡ **Smooth Gameplay** | 60 FPS loop with wall-kick rotation and progressive level speed |

---

## 🧠 How the AI Works

The AI uses a **heuristic evaluation function** — the same family of weights used in classic Tetris AI research.

For every possible **rotation × column** combination, the AI:
1. Simulates dropping the piece into that position
2. Evaluates the resulting board state using 4 features
3. Picks the move with the **highest weighted score**

### Heuristic Weights

| Feature | Weight | Effect |
|---|---|---|
| `lines_cleared` | `+1.0` | Reward for clearing lines |
| `aggregate_height` | `−0.510066` | Penalise tall stacks |
| `holes` | `−0.356630` | Penalise unreachable empty cells |
| `bumpiness` | `−0.184483` | Penalise uneven column heights |

> These weights are derived from the classic Dellacherie / El-Tetris heuristic coefficients.

---

## 🖥️ Screenshots

### Menu Screen
Animated gradient background with glowing **TETRIS AI EDITION** title and two mode buttons.

### Human Mode
Classic Tetris layout with a side panel showing Score, Level, Lines, Next Piece, and Controls.

### AI Mode
Full heuristic dashboard showing:
- Current board stats (Aggregate Height, Holes, Bumpiness)
- Best move selected (rotation, column, score)
- Per-feature breakdown (`raw`, `weight`, `Δ contribution`)
- Heuristic weight bars
- Live column height graph

---

## 🚀 Getting Started

### Prerequisites

- Python **3.8+**
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/tetris-ai.git
cd tetris-ai

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pygame
```

### Run the Game

```bash
python tetris_ai.py
```

---

## 🎮 Controls (Human Mode)

| Key | Action |
|---|---|
| `←` / `→` | Move piece left / right |
| `↑` | Rotate piece |
| `↓` | Soft drop (1 row) |
| `SPACE` | Hard drop (instant) |
| `P` | Pause / Resume |
| `ESC` | Return to main menu |

---

## 📁 Project Structure

```
tetris-ai/
├── tetris_ai.py      # Single-file game — all logic, AI, and rendering
└── README.md
```

All game components are contained within `tetris_ai.py`:

| Class | Responsibility |
|---|---|
| `Board` | Grid state, collision detection, line clearing, heuristic computations |
| `Piece` | Tetromino shape, rotation, movement |
| `AISolver` | Brute-force best-move search using heuristic evaluation |
| `TetrisGame` | Game loop state — score, levels, timing, AI micro-stepping |
| `Renderer` | All Pygame drawing — board, pieces, ghost, panels, overlays |
| `MenuScreen` | Animated main menu with keyboard navigation |

---

## 🔧 Configuration & Tuning

You can tweak constants at the top of `tetris_ai.py`:

```python
# Board dimensions
BOARD_W, BOARD_H = 10, 20

# AI step speed (seconds between each AI micro-move)
self.ai_move_delay = 0.07

# Heuristic weights — experiment to change AI behaviour!
WEIGHTS = {
    'lines_cleared':     1.0,
    'aggregate_height': -0.510066,
    'holes':            -0.35663,
    'bumpiness':        -0.184483,
}
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pygame` | `2.x` | Game window, rendering, input |
| `random` | stdlib | Piece bag / random selection |
| `math` | stdlib | Animated sine-wave effects |
| `time` | stdlib | Frame-independent gravity timing |

---

## 📝 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- Heuristic weights inspired by the **El-Tetris** / **Dellacherie** Tetris AI research
- Built with ❤️ using [Pygame](https://www.pygame.org/)
