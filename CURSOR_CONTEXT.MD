# Project Context — Theodora RL Summative

## Who I am
I am Theodora Egbunike, a final-year Software Engineering student at African Leadership University (ALU) in Kigali, Rwanda. I am the Founder & CEO of Swift Haven Africa (Haven Flow Ltd.), a social enterprise tackling period poverty through IoT-enabled smart sanitary dispensers.

---

## What this project is
This is my RL Summative assignment. I am building a Reinforcement Learning system where an AI agent learns to optimally distribute menstrual pads to schools across a simulated Kigali district. The agent represents a distribution coordinator for a menstrual health NGO (Swift Haven Africa).

**Deadline: April 1, 2026**
**Today: March 26, 2026**
**Days remaining: 6**

---

## The Mission Context
Swift Haven Africa installs IoT-enabled sanitary pad dispensers in low-resource schools across Rwanda. The core logistics problem is: given limited supply and budget, which schools should receive deliveries first? This RL agent learns to solve that problem — prioritizing high-vulnerability, low-stock schools while minimizing travel cost and waste.

---

## Technical Stack
- **RL Library:** stable-baselines3 (SB3)
- **Environment:** Custom Gymnasium environment (gymnasium, not gym)
- **Algorithms:** DQN, REINFORCE, PPO, A2C
- **Visualization:** Three.js (3D browser frontend) + FastAPI (Python backend)
- **Language:** Python 3.10+
- **IDE:** Cursor + Git Bash
- **OS:** Windows 11
- **Version Control:** Git via SSH (HTTPS blocked by Windows Defender)

---

## Project Structure
```
theodora-rl-summative/
├── environment/
│   ├── custom_env.py         # Custom Gymnasium environment
│   └── rendering.py          # FastAPI server + visualization bridge
├── training/
│   ├── dqn_training.py       # DQN training + 10 hyperparameter runs
│   └── pg_training.py        # REINFORCE, PPO, A2C training + 10 runs each
├── models/
│   ├── dqn/                  # Saved DQN models
│   └── pg/                   # Saved policy gradient models
├── visualization/
│   ├── index.html            # Three.js 3D frontend
│   └── main.js               # Three.js scene logic
├── main.py                   # Entry point — loads best model, runs with viz
├── requirements.txt          # All dependencies
└── README.md
```

---

## Environment Design (custom_env.py)

### Problem
Agent must decide which school to deliver pads to at each timestep across a 4-week episode.

### Schools (12 total, based on Kigali districts)
Each school has:
- Name and district
- x, y coordinates (for map positioning)
- Vulnerability score (0.0–1.0) — based on poverty index + rural remoteness
- Weekly demand (number of pads needed)
- Current stock level (changes each step)
- Distance from depot (km)

### Action Space
- **Type:** Discrete
- **Size:** 12 (one action per school = deliver to that school)
- Agent picks one school to deliver to per timestep

### Observation Space
- **Type:** Box (continuous)
- **Shape:** (12, 4) flattened to (48,)
- Per school: [stock_level, weekly_demand, distance_from_depot, vulnerability_score]
- All values normalized to [0, 1]

### Reward Function
```
+10  → delivering to a school with stock < 20% capacity (high need)
+5   → any successful delivery
-2   → per unit of normalized distance traveled (efficiency penalty)
-5   → delivering to a school already well-stocked (> 80% capacity)
-1   → per timestep (encourages efficiency)
+20  → episode bonus if all high-vulnerability schools (score > 0.7) are covered
```

### Start State
- Depot: 500 pads fully stocked
- All schools: randomized low stock (0–30% capacity)
- Week: 1, Step: 0

### Terminal Conditions
- All 500 pads distributed
- 4 weeks (28 steps) elapsed
- All schools reach adequate stock (> 60% capacity)

### Episode Length
- Max 28 steps (4 weeks × 7 days)

---

## Visualization Design (Three.js + FastAPI)

### Architecture
```
Python RL Agent (SB3)
        ↓ calls
FastAPI server (rendering.py) — serves /state and /step endpoints
        ↓ websocket/polling
Three.js frontend (index.html) — 3D browser visualization at localhost:8080
```

### Visual Elements
- Flat terrain representing Kigali geography
- 12 school buildings (low-poly 3D boxes) positioned on map
- School color = stock level (red = critical, amber = low, green = adequate)
- School size = vulnerability score (larger = more vulnerable)
- Delivery vehicle (3D box/truck) animating between schools
- Depot node at center
- HUD overlay showing: Episode, Step, Current Action, Total Reward, Speed

### Reference
The target visual quality is similar to the "Daladala Agent - Autonomous Transit" project — a browser-based 3D RL visualization with debug controls panel, reward display, and camera views.

---

## Training Requirements

### For each algorithm, run 10 hyperparameter combinations:

**DQN columns:** Learning Rate | Gamma | Replay Buffer Size | Batch Size | Exploration (epsilon) | Mean Reward

**REINFORCE columns:** Learning Rate | Gamma | Entropy Coef | N Steps | Mean Reward

**PPO columns:** Learning Rate | Clip Range | N Steps | Entropy Coef | N Epochs | Mean Reward

**A2C columns:** Learning Rate | Gamma | N Steps | VF Coef | Entropy Coef | Mean Reward

### Training length per run
- 50,000 timesteps per run (keeps each run under 10 minutes)
- Save best model based on mean reward over last 10 episodes

---

## Report Requirements
- **Length:** 7–10 pages, saved as PDF
- **Citation style:** Follow report template
- **Required plots:**
  - Cumulative reward curves (all 4 methods in subplots)
  - DQN objective/loss curves
  - PG entropy curves (REINFORCE, PPO, A2C)
  - Convergence plots (episodes to stable performance)
  - Generalization test results (unseen initial states)

---

## Rubric Summary
| Criterion | Points | Key requirement |
|---|---|---|
| Environment Validity & Complexity | 10 | Rich env, exhaustive action space, meaningful rewards |
| Hyperparameter Experiments | 10 | 4 tables × 10 rows, varied params, explained behavior |
| System Implementation & Agent Behavior | 10 | Advanced 3D viz, agent behavior aligns with goal, JSON API to frontend |
| Discussion & Analysis | 10 | All plots included, labelled, discussed with depth |
| Video Demonstration | 5 | Camera on, full screen, all rubric points stated verbally |

---

## Rules for Cursor when helping with this project

1. **Always use `gymnasium` not `gym`** — SB3 requires gymnasium
2. **Environment must pass `stable_baselines3.common.env_checker.check_env()`** before any training
3. **All 4 algorithms must use the exact same custom environment** — no modifications between algorithms
4. **Keep episodes short** (max 28 steps) so 50k timestep runs complete quickly
5. **FastAPI server must be non-blocking** — RL training loop and visualization must run concurrently
6. **Three.js must run in browser at localhost:8080** — no external dependencies beyond CDN
7. **requirements.txt must be complete** — project will be cloned and executed by the lecturer
8. **Never use deprecated gym.spaces** — use gymnasium.spaces throughout
9. **Save models with `.zip` extension** using `model.save()` from SB3
10. **All hyperparameter runs must log mean_reward** to a results dict or CSV for table generation

---

## Current Status (March 26, 2026)
- [x] GitHub repo created: `theodora-rl-summative`
- [x] SSH authentication configured
- [x] Folder structure created in Cursor
- [ ] custom_env.py — IN PROGRESS
- [ ] check_env() passing
- [ ] rendering.py (FastAPI)
- [ ] visualization/ (Three.js)
- [ ] dqn_training.py
- [ ] pg_training.py
- [ ] main.py
- [ ] All 40 training runs complete
- [ ] All plots generated
- [ ] Report written
- [ ] Video recorded
- [ ] Submitted to Canvas

---

## How to use this context
Paste this entire block at the start of any Cursor AI chat session. When asking Cursor for code, reference the specific file you're working on (e.g. "help me complete custom_env.py based on the spec above"). Cursor will have full context of the environment design, reward function, obs/action spaces, and project structure without you needing to re-explain every time.