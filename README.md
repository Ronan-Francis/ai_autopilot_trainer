# Neural Network Autopilot for Tunnel Navigation

## Overview

This project implements a real-time neural network autopilot for a 2D tunnel navigation game. The system uses a staged training approach that transitions from a rule-based "teacher" to a neural network capable of flying autonomously. All data collection, training, evaluation, and decision-making occur within the game loop, with a target performance of 30+ seconds of autonomous flight.

---

## Feature Engineering and Data Design

Instead of using the full 30×20 grid (600 inputs), a focused 10-dimensional feature vector was engineered:

- Distance to obstacles directly above and below the plane
- Average forward-looking distances (top and bottom)
- Player’s vertical position
- Tunnel center, width, and movement trend
- Rate of change in tunnel edges above and below

All features are normalized to [0, 1] to simplify training. Data is collected during gameplay and labeled based on the current action (UP, STAY, DOWN). Samples immediately before crashes are discarded to prevent learning from bad decisions. Only successful flight segments are saved permanently to improve generalization and avoid overfitting.

---

## Neural Network Design

- **Type**: Classification (UP, STAY, DOWN)
- **Input Layer**: 10 neurons
- **Hidden Layers**: 3 layers — 20 → 10 → 5 neurons
- **Output Layer**: 3 neurons with Softmax activation
- **Activation Functions**: TANH for hidden layers, Softmax for output

### Rationale

This compact architecture ensures fast training (< 1 minute) and stable inference. The bottleneck design encourages abstraction. Softmax output enables confidence-based decision-making. TANH works well with normalized inputs and avoids vanishing gradient issues.

---

## Training Strategy

Training is performed using **Resilient Propagation (RProp)** with the following parameters:

| Parameter            | Value     | Purpose                                           |
|----------------------|-----------|---------------------------------------------------|
| Max Epochs           | 5000      | Allows convergence without excessive runtime      |
| Target Error         | 0.008     | Ensures sufficient accuracy                       |
| L2 Regularization    | 0.0003    | Prevents overfitting to short-term patterns       |
| Training Timeout     | 50s       | Keeps training within real-time limits            |
| Minimum Samples      | 20        | Avoids premature training with weak data          |

Models are automatically saved (`resources/network_model.eg`) and reloaded when available.

---

## Progressive Control System

The control system evolves in three stages:

1. **Teacher-Only**: Rule-based controller manages flight; NN observes and trains.
2. **Hybrid**: NN makes decisions; teacher intervenes when confidence is low or danger is detected.
3. **Autonomous**: NN flies independently. System reverts to earlier stages on failure or low confidence.

### Progression Triggers

- Confidence ≥ 60%
- 30+ seconds of successful autonomous flight
- Reversion triggered by crashes or erratic behavior

This progression ensures safe, stable training and guards against poor generalization.

---

## Rule-Based Teacher

The teacher system uses a clear rule set:

- Avoid walls when within 2 units
- Center the plane vertically when off-course
- Anticipate tunnel trends (up/down)
- Override actions that may lead to collisions

These rules provide consistent, interpretable training signals for the neural network.

---

## Integration and Controls

All functionality is integrated into the game. Training, monitoring, and decision control are accessible via key bindings:

- `T`: Trigger training
- `C`: Clear collected data
- `F2`: Toggle autopilot monitor
- `F3`: Toggle teacher panel
- `F4`: Toggle flight analyzer
- `D`, `G`, `V`: Toggle debug info, confidence graph, and visual overlays

---

## Monitoring and Evaluation

Multiple in-game tools support evaluation and debugging:

- **Autopilot Monitor**: Displays live confidence scores and NN status
- **Teacher Panel**: Compares NN and rule-based decisions
- **Flight Analyzer**: Logs crash rates, flight duration, and decision quality
- **Game View Overlay**: Displays mode, actions, and confidence in real time

Logs are stored in `logs/game.log` for review.

---

## Deployment

- Executable: `ai.jar`
- Run with:  
  ```bash
  java -cp ./ai.jar ie.atu.sw.Runner
- Resources:
	- resources/network_model.eg: Serialized model
	- resources/game.properties: Configuration
- No hardcoded paths or IDE dependencies used

Summary
This project combines compact feature engineering, real-time training, and a progressive control structure to build a functional and interpretable autopilot system. 
The design emphasizes fast convergence, safe behavior, and full runtime integration. 
Monitoring tools and a modular architecture allow for both experimentation and evaluation, meeting all requirements of the assignment while remaining extensible.