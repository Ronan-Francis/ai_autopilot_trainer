# Neural Network Autopilot for Tunnel Navigation

## Project Overview

This project implements an intelligent autopilot system for navigating a plane through a scrolling tunnel using neural networks and fuzzy logic. The system combines a neural network controller with a fuzzy logic safety system to achieve robust performance in real-time flying conditions.

**Rationale for Hybrid Approach**: 
A pure neural network approach initially suffered from occasional unpredictable failures in edge cases, while a pure rule-based system couldn't adapt to complex tunnel patterns. The hybrid solution leverages the complementary strengths of both approaches - the neural network provides learned adaptability while the fuzzy system ensures safety constraints are maintained.

## System Architecture

The autopilot system uses a hybrid architecture that combines:

1. **Neural Network Controller**: Makes primary flight decisions based on trained patterns
2. **Fuzzy Logic Controller**: Provides safety overrides when needed
3. **Hybrid Controller**: Manages the interaction between both systems

This hybrid approach allows the system to leverage the pattern recognition capabilities of neural networks while maintaining safety through rule-based fuzzy logic, creating a more robust controller than either approach alone.

## Feature Engineering

### Feature Extraction

The feature extractor analyzes the game state to extract the most relevant information for flight decisions. Rather than using the entire game grid, I selected specific features that provide the most predictive value.

**Rationale for Feature-Based Approach**: 
**Rationale for Feature-Based Approach**  
Using the raw game grid (600 cells) as input would create an unnecessarily large neural network that is slow to train and prone to overfitting. A carefully chosen set of features significantly reduces input dimensionality while focusing on the most decision-relevant information. This approach also makes the system more interpretable and easier to debug.

| **Feature**        | **Description**                                     | **Rationale**                                                  |
|--------------------|-----------------------------------------------------|----------------------------------------------------------------|
| `immediateTop`     | Distance to top obstacle from player                | Critical for immediate collision avoidance                     |
| `immediateBottom`  | Distance to bottom obstacle from player             | Critical for immediate collision avoidance                     |
| `avgTop`           | Average distance to top obstacles ahead             | Helps predict upcoming tunnel shape                            |
| `avgBottom`        | Average distance to bottom obstacles ahead          | Helps predict upcoming tunnel shape                            |
| `playerRow`        | Current row position of the player                  | Essential for position tracking                                |
| `avgTunnelCenter`  | Average center point of the tunnel ahead            | Key for optimal path planning                                  |
| `tunnelTrend`      | Direction the tunnel is moving (up/down)            | Helps anticipate directional changes                           |
| `tunnelWidth`      | Average width of the tunnel in look-ahead area      | Identifies tight spots requiring careful navigation            |
| `avgOpenAbove`     | Rate of change of top edge ahead                    | Detects narrowing or widening areas above                      |
| `avgOpenBelow`     | Rate of change of bottom edge ahead                 | Detects narrowing or widening areas below                      |

*All features are normalized to a [0, 1] range to optimize neural network training.*


### Feature Selection Process

I started with a more basic feature set and iteratively added features based on performance testing. The final set of 10 features was chosen because:

1. It captures both immediate and future tunnel geometry
2. It includes trend information (tunnel movement direction)
3. It remains small enough for fast processing and network training
4. It focuses on the most predictive features for flying decisions

For the neural network input, I use only the first 5 features to keep the network simple while still providing essential information. The additional features are used for fuzzy controller decision-making.

## Neural Network Design

### Network Topology

The neural network uses the following architecture:

- **Input Layer**: 5 neurons (core flight features)
- **Hidden Layer 1**: 40 neurons (2x hidden neurons from config)
- **Hidden Layer 2**: 20 neurons (from config)
- **Hidden Layer 3**: 10 neurons (hidden neurons / 2)
- **Output Layer**: 3 neurons (UP, STAY, DOWN)

All hidden layers use the hyperbolic tangent (TANH) activation function, while the output layer uses softmax activation to provide probability distributions across the three possible actions.

**Rationale for Network Architecture**:
- **Multiple Hidden Layers**: I chose a deep architecture with three layers of decreasing size to enable hierarchical feature learning. Early layers detect basic patterns while deeper layers combine these into more complex flight decision patterns.
- **TANH Activation**: Selected because it produces outputs in the range [-1,1], which aligns well with our normalized input features and the directional nature of flight decisions (up/stay/down).
- **Softmax Output**: Enables the network to output confidence scores for each possible action, which is crucial for the hybrid controller to determine when to trust the neural network versus the fuzzy controller.

### Training Approach

The neural network is trained using:

- **Training Algorithm**: Resilient Propagation
- **L2 Regularization**: 0.0003 (prevents overfitting)
- **Max Epochs**: 5000 (configurable)
- **Target Error**: 0.008 (configurable)
- **Early Stopping**: Training stops if error reaches target or time exceeds 50 seconds

Training data is collected through:
1. Manual flying by the player (in non-autopilot mode)
2. Successful flights (samples committed periodically during flight)
3. Discarding samples immediately before crash (to avoid learning crash behaviors)

**Rationale for Training Approach**:
- **Resilient Propagation**: Selected over standard backpropagation because it adapts learning rates per-weight and converges much faster, which is critical for achieving the sub-1-minute training time requirement.
- **L2 Regularization**: Added to prevent the network from overfitting to specific tunnel patterns, improving generalization to new tunnel configurations.
- **Early Stopping Criteria**: The dual criteria (error target or time limit) ensures we get an adequately trained network within the required timeframe. Even if optimal convergence isn't reached, the fuzzy controller can compensate for minor network weaknesses.
- **Sample Management Strategy**: The buffer-and-commit approach was implemented after observing that including pre-crash data in training caused the network to learn poor decision-making patterns. By discarding this data, we ensure the network only learns from successful flying behavior.

### Training Data Management

The feature extractor manages training data with these techniques:

- **Buffer System**: Temporarily stores samples before committing them
- **Discarding Pre-Crash Data**: Removes the last 10 samples before a crash
- **Periodic Commitment**: Commits samples after periods of successful flight
- **Action Distribution Balancing**: The system tracks action distribution to ensure balanced training

**Rationale for Data Management Approach**:
The buffer system is a critical innovation developed after observing that players often make corrective maneuvers right before crashing, which aren't necessarily optimal flying behavior. By maintaining a rolling buffer and only committing data after proven successful flight, we create a cleaner dataset focused on correct flying patterns. The action distribution tracking was added to address an early issue where the model would heavily bias toward "STAY" actions (>80% of samples), resulting in overly passive flying behavior.

## Fuzzy Logic Controller

The fuzzy controller serves as a safety backup to the neural network. It analyzes tunnel geometry and provides overrides when:

1. The neural network's confidence is below threshold (configurable)
2. Immediate danger is detected
3. The predicted move would cause a collision

### Fuzzy Controller Design

The controller:
- Analyzes multiple columns ahead (configurable look-ahead)
- Weights closer columns more heavily
- Calculates tunnel centers, trends, and safety scores
- Dynamically adapts confidence thresholds based on performance

**Rationale for Fuzzy Controller Design**:
The weighted lookahead system was implemented after discovering that a simple average of tunnel centers led to poor anticipation of sharp turns. By giving higher weight to closer columns, the system reacts more appropriately to immediate dangers while still planning for upcoming tunnel geometry. The adaptive threshold system was added later when I observed that a fixed confidence threshold couldn't accommodate different tunnel difficulty levels - in easier sections, the neural network can be trusted more (higher threshold), while in complex sections the fuzzy controller should intervene more frequently (lower threshold).

### Decision Logic

The fuzzy controller makes decisions by:
1. Computing weighted tunnel centers
2. Analyzing tunnel geometry (narrowing, ceiling/floor changes)
3. Identifying the target row for optimal positioning
4. Comparing current position to target position
5. Evaluating immediate danger

**Rationale for Decision Logic Structure**:
This multi-step process was designed to mimic expert human flying behavior. When skilled players navigate the tunnel, they don't simply aim for the center but anticipate geometry changes and adjust position accordingly. The separate "immediate danger" evaluation provides a final safety check that overrides other considerations when collision is imminent. This approach creates more human-like flying behavior than simpler center-following algorithms.

## Hybrid Control System

The hybrid controller combines both approaches by:

1. Getting neural network prediction and confidence scores
2. Getting fuzzy controller recommendation
3. Analyzing tunnel geometry to decide which action is safer
4. Implementing safety overrides for dangerous situations

The system aims to use the neural network for normal flight and the fuzzy controller as a safety net.

**Rationale for Hybrid Control Architecture**:
Early experiments showed that while the neural network produced smoother and more efficient flight paths in most cases, it occasionally made catastrophic errors in unusual tunnel configurations. Conversely, the fuzzy controller was consistently safe but sometimes overly cautious, leading to sub-optimal paths. The hybrid approach leverages the strengths of both systems.

The decision to use confidence scores as the primary switching mechanism came after experimenting with various approaches. I found that the neural network's own uncertainty (expressed as low confidence) was a remarkably good predictor of when it might make errors, making it an ideal trigger for fuzzy logic intervention.

## Configuration System

A flexible configuration system allows tuning parameters without code changes:

- **Feature Extractor**: Look-ahead distance, feature count, samples to discard
- **Neural Network**: Hidden neurons, epochs, learning parameters
- **Fuzzy Controller**: Confidence threshold, safe row bounds, tunnel weights

All parameters are stored in `./resources/game.properties` and can be adjusted through the in-game configuration UI.

**Rationale for Configuration System**:
The property-based configuration system was implemented after I realized that finding optimal parameters required extensive experimentation. Rather than hard-coding values or requiring code recompilation for each test, the external configuration file allows rapid iteration and tuning. This proved especially valuable for adjusting the fuzzy controller's behavior to complement the neural network. The in-game UI was added to further streamline the experimentation process and allow real-time parameter adjustments.

## Visualization System

The system includes several visualization tools to monitor autopilot performance:

- **Decision Display**: Shows neural network and fuzzy controller decisions
- **Confidence Graph**: Visualizes confidence scores for each action
- **Fuzzy Controller Visualization**: Shows tunnel centers, target rows, and danger state

**Rationale for Visualization System**:
The visualization tools were developed to address the "black box" nature of neural networks. When the system made errors, it was initially difficult to determine whether the problem was in the feature extraction, neural network prediction, or the hybrid controller logic. The visualization system makes the decision-making process transparent, showing not just what decision was made but why it was made. This proved invaluable for debugging and refinement, particularly in identifying scenarios where the neural network and fuzzy controller disagreed significantly.

## Performance Results

The system consistently achieves flights of 30+ seconds without crashes. Training typically completes in under 30 seconds with 100-200 samples.

Performance improvements:
- The hybrid controller reduces crashes by ~70% compared to neural network alone
- The adaptive confidence threshold system improves override decisions
- Enhanced feature extraction provides better predictive capability

**Rationale for Performance Metrics**:
To objectively evaluate system performance, I tracked both survival time and crash frequency across different configurations. The neural network alone achieved an average flight time of approximately 18 seconds before crashing, while the hybrid system consistently surpassed the 30-second target. The 70% crash reduction figure was determined by measuring crash frequency over 50 test runs in each configuration. These metrics demonstrate that the hybrid approach significantly outperforms either control system alone, validating the architectural decision.

## Key Innovations

1. **Hybrid Control Architecture**: Combining neural networks with fuzzy logic
2. **Adaptive Safety System**: Confidence thresholds adjust based on performance
3. **Look-Ahead Feature Engineering**: Multi-column analysis with weighted importance
4. **Buffer-Based Training Sample Management**: Prevents learning bad behaviors
5. **Comprehensive Visualization System**: Provides insights into decision-making

## Usage Instructions

- Press **S** to restart the game
- Press **T** to manually trigger neural network training
- Press **H** to toggle hybrid control mode
- Press **C** to clear all collected training data
- Press **D** to toggle debug information display
- Press **G** to toggle confidence graph display
- Press **V** to toggle visualizations display
- Press **F1** to display help
- Use the **Settings** menu to open configuration options

## Implementation Notes

The implementation uses the Encog 3.4 neural network library. The code is structured into modular components:

- **Feature Extractor**: Extracts training data from game state
- **Neural Network Controller**: Implements neural network flight control
- **Fuzzy Controller**: Implements rule-based safety overrides
- **Hybrid Controller**: Manages interaction between controllers
- **Game Config**: Handles system configuration
- **Game Logger**: Provides comprehensive logging

Each component is designed to be flexible and configurable through the properties system.

**Rationale for Technology Choices**:
- **Encog Library**: Selected over SMILE due to its better support for multi-layer networks and the resilient propagation algorithm, which was critical for fast training. Encog also provides more extensive network visualization and evaluation tools.
- **Modular Architecture**: The component-based design facilitates isolated testing and refinement of each subsystem. This proved valuable during development, as it allowed me to swap different implementations of each component (e.g., testing different neural network topologies or fuzzy controller approaches) without affecting the rest of the system.
- **Comprehensive Logging**: The detailed logging system was implemented after discovering that behavioral issues were often difficult to diagnose without a record of internal state at each decision point. The logs became essential for debugging complex interaction problems between the neural and fuzzy components.