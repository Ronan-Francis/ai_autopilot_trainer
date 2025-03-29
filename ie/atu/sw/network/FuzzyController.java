package ie.atu.sw.network;

import java.util.LinkedList;

/**
 * An improved fuzzy logic controller that provides safety overrides but is more
 * conservative in when it intervenes.
 */
public class FuzzyController {
	// Constants
	private static int LOOK_AHEAD;
	private static final int PLAYER_COLUMN = 15;
	private static final int MODEL_HEIGHT = 20;

	// Lower confidence threshold to reduce overrides
	private double confidenceThreshold = 0.65; // Increased from 0.5 to 0.65

	// Safety margins
	private static int MIN_SAFE_ROW;
	private static int MAX_SAFE_ROW;

	// Pattern recognition weights
	private static double[] TUNNEL_WEIGHTS; // More weight on closest column

	// Performance tracking
	private int successfulMoves = 0;
	private int overriddenMoves = 0;
	private double[] recentPerformance = new double[20];
	private int performanceIndex = 0;

	// Last calculated values (for logging)
	private int[] lastTunnelCenters = new int[LOOK_AHEAD];
	private int lastTargetRow;
	private boolean lastDangerState;

	public FuzzyController() {
		// Load configuration
		GameConfig config = GameConfig.getInstance();
		LOOK_AHEAD = config.getLookAhead();
		confidenceThreshold = config.getConfidenceThreshold();
		MIN_SAFE_ROW = config.getMinSafeRow();
		MAX_SAFE_ROW = config.getMaxSafeRow();
		TUNNEL_WEIGHTS = config.getTunnelWeights();

		// Initialize last tunnel centers array with the correct size
		lastTunnelCenters = new int[LOOK_AHEAD];

		GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FUZZY_CONTROLLER,
				"Fuzzy Controller initialized with configuration: " + "lookAhead=" + LOOK_AHEAD + ", "
						+ "confidenceThreshold=" + confidenceThreshold);
	}

	/**
	 * Determines the action to take using improved fuzzy logic based on tunnel centers,
	 * geometry trends, and player position.
	 *
	 * @param model     The game grid
	 * @param playerRow The current row position of the player
	 * @return The final action (-1=up, 0=stay, 1=down)
	 */
	public int determineAction(LinkedList<byte[]> model, int playerRow) {
	    // Calculate tunnel centers and geometry for the next few columns
	    int[] tunnelCenters = new int[LOOK_AHEAD];
	    int[] tunnelTops = new int[LOOK_AHEAD];
	    int[] tunnelBottoms = new int[LOOK_AHEAD];
	    int[] tunnelWidths = new int[LOOK_AHEAD];
	    
	    for (int i = 0; i < LOOK_AHEAD; i++) {
	        int columnIndex = (PLAYER_COLUMN + i + 1) % model.size();
	        byte[] column = model.get(columnIndex);

	        // Find the top edge of the tunnel
	        int topEdge = 0;
	        while (topEdge < MODEL_HEIGHT && column[topEdge] != 0) {
	            topEdge++;
	        }
	        tunnelTops[i] = topEdge;

	        // Find the bottom edge of the tunnel
	        int bottomEdge = MODEL_HEIGHT - 1;
	        while (bottomEdge >= 0 && column[bottomEdge] != 0) {
	            bottomEdge--;
	        }
	        tunnelBottoms[i] = bottomEdge;

	        // Calculate tunnel center and width
	        tunnelCenters[i] = (topEdge + bottomEdge) / 2;
	        tunnelWidths[i] = (bottomEdge - topEdge + 1);
	    }

	    // Store for logging
	    this.lastTunnelCenters = tunnelCenters.clone();

	    // Calculate trends in tunnel geometry
	    double centerTrend = 0;
	    double topTrend = 0;
	    double bottomTrend = 0;
	    
	    for (int i = 1; i < LOOK_AHEAD; i++) {
	        centerTrend += tunnelCenters[i] - tunnelCenters[i-1];
	        topTrend += tunnelTops[i] - tunnelTops[i-1];
	        bottomTrend += tunnelBottoms[i] - tunnelBottoms[i-1];
	    }
	    
	    // Normalize trends
	    centerTrend /= (LOOK_AHEAD - 1);
	    topTrend /= (LOOK_AHEAD - 1);
	    bottomTrend /= (LOOK_AHEAD - 1);
	    
	    // Log trends for analysis
	    GameLogger.log(GameLogger.LogLevel.DEBUG, GameLogger.LogCategory.FUZZY_CONTROLLER,
	                  String.format("Tunnel trends: Center=%.2f, Top=%.2f, Bottom=%.2f", 
	                               centerTrend, topTrend, bottomTrend));

	    // Compute weighted average of tunnel centers with more weight on closer columns
	    double weightedSum = 0;
	    double totalWeight = 0;
	    for (int i = 0; i < LOOK_AHEAD; i++) {
	        // Use TUNNEL_WEIGHTS safely by taking the min of array length and index
	        double weight = (i < TUNNEL_WEIGHTS.length) ? TUNNEL_WEIGHTS[i] : TUNNEL_WEIGHTS[TUNNEL_WEIGHTS.length - 1];
	        weightedSum += tunnelCenters[i] * weight;
	        totalWeight += weight;
	    }
	    int targetRow = (int) Math.round(weightedSum / totalWeight);
	    
	    // Store target row for logging
	    this.lastTargetRow = targetRow;

	    // Analyze tunnel geometry to predict best path
	    boolean tunnelNarrowing = false;
	    boolean ceilingLowering = false;
	    boolean floorRising = false;
	    
	    // Check if tunnel is narrowing
	    double avgWidth = 0;
	    for (int i = 0; i < LOOK_AHEAD; i++) {
	        avgWidth += tunnelWidths[i];
	    }
	    avgWidth /= LOOK_AHEAD;
	    
	    // Detect geometry changes
	    if (tunnelWidths[0] > avgWidth) tunnelNarrowing = true;
	    if (topTrend > 0) ceilingLowering = true;
	    if (bottomTrend < 0) floorRising = true;
	    
	    // Keep target row in safe bounds
	    targetRow = Math.max(MIN_SAFE_ROW, Math.min(MAX_SAFE_ROW, targetRow));

	    // Check for immediate danger
	    boolean inDanger = isInDanger(model, playerRow);
	    this.lastDangerState = inDanger;

	    // Improved decision logic with geometry analysis
	    int action;
	    int rowDifference = playerRow - targetRow;

	    if (inDanger) {
	        // In danger - make more decisive moves
	        if (rowDifference > 0) {
	            action = -1; // Move up
	        } else if (rowDifference < 0) {
	            action = 1; // Move down
	        } else {
	            // At target but still in danger, analyze tunnel geometry for best escape
	            if (ceilingLowering && !floorRising) {
	                action = 1; // Ceiling coming down, move down
	            } else if (floorRising && !ceilingLowering) {
	                action = -1; // Floor coming up, move up
	            } else {
	                // Both or neither - choose based on available space
	                double topSpace = tunnelTops[0];
	                double bottomSpace = MODEL_HEIGHT - tunnelBottoms[0];
	                
	                if (topSpace > bottomSpace) {
	                    action = -1; // More space above, move up
	                } else {
	                    action = 1; // More space below, move down
	                }
	            }
	        }
	    } else {
	        // Not in danger - more conservative movement
	        if (Math.abs(rowDifference) <= 1) {
	            action = 0; // Stay if close enough
	        } else if (rowDifference > 0) {
	            action = -1; // Move up
	        } else {
	            action = 1; // Move down
	        }
	        
	        // Override if tunnel geometry suggests a different action
	        if (action == 0) {
	            // If staying but tunnel geometry suggests a move would be safer
	            if (centerTrend < -1.5) { // Tunnel center moving up significantly
	                action = -1; // Move up to follow
	                GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FUZZY_CONTROLLER,
	                              "Geometry override: Tunnel moving UP, following");
	            } else if (centerTrend > 1.5) { // Tunnel center moving down significantly
	                action = 1; // Move down to follow
	                GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FUZZY_CONTROLLER,
	                              "Geometry override: Tunnel moving DOWN, following");
	            }
	        }
	    }

	    // Log fuzzy controller decision with enhanced geometry analysis
	    String geometryInfo = String.format("Center trend: %.1f, Top trend: %.1f, Bottom trend: %.1f", 
	                                      centerTrend, topTrend, bottomTrend);
	    GameLogger.logFuzzyControllerDecision(playerRow, lastTunnelCenters, lastTargetRow, action, lastDangerState);
	    GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FUZZY_CONTROLLER,
	                  "Geometry analysis: " + geometryInfo);

	    // Update performance metrics
	    updatePerformanceMetrics(model, playerRow, action);

	    return action;
	}
	/**
	 * Checks if the fuzzy controller should override the neural network's decision.
	 * More conservative than previous version - only overrides in clear danger
	 * situations.
	 *
	 * @param model            The game grid
	 * @param playerRow        The current player row
	 * @param nnAction         The action predicted by the neural network
	 * @param nnConfidence     The confidence score for the NN's predicted action
	 * @param confidenceScores Array of confidence scores for all actions
	 * @return The overriding action to take, or null if no override is needed
	 */
	public Integer checkForOverride(LinkedList<byte[]> model, int playerRow, int nnAction, double nnConfidence,
			double[] confidenceScores) {
		// More strict override conditions
		boolean predictedMoveDangerous = wouldMoveCauseDanger(model, playerRow, nnAction);

		// Lower the threshold to trigger overrides more frequently
		boolean lowConfidence = nnConfidence < 0.75; // Increased from 0.65

		// Get fuzzy controller's suggested action
		int fuzzyAction = determineAction(model, playerRow);

		// Override in more situations
		if (predictedMoveDangerous || (lowConfidence && fuzzyAction != nnAction) || isInDanger(model, playerRow)) {

			overriddenMoves++;
			GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FUZZY_CONTROLLER,
					"Override triggered - Fuzzy suggests: " + actionToString(fuzzyAction));

			// In most cases, use the fuzzy controller's action directly
			return fuzzyAction;
		}

		// No override needed
		return null;
	}

	private String actionToString(int action) {
		switch (action) {
		case -1 -> {
			return "UP";
		}
		case 0 -> {
			return "STAY";
		}
		case 1 -> {
			return "DOWN";
		}
		default -> {
			throw new IllegalArgumentException("Unexpected value: " + action);
		}
		}
	}

	/**
	 * Checks if the player is in immediate danger. This is public so it can be
	 * called from manual mode.
	 * 
	 * @param model     The game grid
	 * @param playerRow The current row position of the player
	 * @return true if there's significant danger in the player's path
	 */
	public boolean isInDanger(LinkedList<byte[]> model, int playerRow) {
		// Check for immediate obstacles in the next 2 columns
		for (int ahead = 1; ahead <= 2; ahead++) {
			int columnIndex = (PLAYER_COLUMN + ahead) % model.size();
			byte[] column = model.get(columnIndex);

			// Check player row and adjacent rows
			if (playerRow >= 0 && playerRow < MODEL_HEIGHT && column[playerRow] != 0) {
				return true; // Obstacle in direct path
			}

			if (ahead == 1) { // For the immediate next column, also check adjacent cells
				if (playerRow > 0 && column[playerRow - 1] != 0) {
					return true; // Obstacle above
				}
				if (playerRow < MODEL_HEIGHT - 1 && column[playerRow + 1] != 0) {
					return true; // Obstacle below
				}
			}
		}

		// Also check if player is too close to edges
		return playerRow < MIN_SAFE_ROW || playerRow > MAX_SAFE_ROW;
	}

	/**
	 * Checks if a move would put the player in immediate danger.
	 *
	 * @param model     The game grid
	 * @param playerRow The current player row
	 * @param action    The move action (-1, 0, 1)
	 * @return true if the move is dangerous, false otherwise
	 */
	private boolean wouldMoveCauseDanger(LinkedList<byte[]> model, int playerRow, int action) {
		int newRow = playerRow + action;

		// Check bounds
		if (newRow < 1 || newRow >= MODEL_HEIGHT - 1) {
			return true; // Consider very close to edge as dangerous
		}

		// Check next 2 columns for collision
		for (int ahead = 1; ahead <= 2; ahead++) {
			int columnIndex = (PLAYER_COLUMN + ahead) % model.size();
			byte[] column = model.get(columnIndex);

			// Check player position and adjacent positions for obstacles
			if (column[newRow] != 0) {
				return true; // Direct collision path
			}

			// For the immediate next column, also check one row above and below
			if (ahead == 1) {
				if (column[newRow - 1] != 0 || column[newRow + 1] != 0) {
					return true; // Very close to obstacle
				}
			}
		}

		// Check if moving would place us very close to a wall
		// Look at the next column
		int nextColumnIndex = (PLAYER_COLUMN + 1) % model.size();
		byte[] nextColumn = model.get(nextColumnIndex);

		// Find the top and bottom edges of the tunnel
		int topEdge = 0;
		while (topEdge < MODEL_HEIGHT && nextColumn[topEdge] == 0) {
			topEdge++;
		}

		int bottomEdge = MODEL_HEIGHT - 1;
		while (bottomEdge >= 0 && nextColumn[bottomEdge] == 0) {
			bottomEdge--;
		}

		// If new position would be very close to top or bottom edge, consider it
		// dangerous
		if (newRow - topEdge <= 1 || bottomEdge - newRow <= 1) {
			return true;
		}

		return false;
	}

	/**
	 * Calculates a safety score for a given row position.
	 * 
	 * @param model The game grid
	 * @param row   The row to evaluate
	 * @return A safety score (higher is safer)
	 */
	private double calculateSafetyScore(LinkedList<byte[]> model, int row) {
		// Invalid row has lowest score
		if (row < 0 || row >= MODEL_HEIGHT) {
			return -1000;
		}

		double score = 100.0; // Base score

		// Check more columns ahead (increased from 3 to 4)
		for (int ahead = 1; ahead <= 4; ahead++) {
			int columnIndex = (PLAYER_COLUMN + ahead) % model.size();
			byte[] column = model.get(columnIndex);

			// Higher weight for closer columns
			double weight = 1.0 / Math.sqrt(ahead); // Changed from 1.0/ahead to give more weight to farther columns

			// Check for obstacles in and around row
			if (column[row] != 0) {
				score -= 70 * weight; // Increased penalty for direct obstacle
			}

			// Check more adjacent rows for better path planning
			for (int offset = 1; offset <= 2; offset++) {
				if (row - offset >= 0 && column[row - offset] != 0) {
					score -= 20 * weight / offset;
				}
				if (row + offset < MODEL_HEIGHT && column[row + offset] != 0) {
					score -= 20 * weight / offset;
				}
			}

			// Bonus for clear path
			if (column[row] == 0) {
				// Check if the cell is in the middle of an open area
				int clearCellsAbove = 0;
				for (int i = row - 1; i >= 0 && column[i] == 0; i--)
					clearCellsAbove++;

				int clearCellsBelow = 0;
				for (int i = row + 1; i < MODEL_HEIGHT && column[i] == 0; i++)
					clearCellsBelow++;

				// Reward being in the center of open areas
				// Reward being in the center of open areas
				int openSpace = clearCellsAbove + clearCellsBelow + 1;
				score += 10 * weight * Math.min(clearCellsAbove, clearCellsBelow)
						/ Math.max(1, Math.abs(clearCellsAbove - clearCellsBelow) + 1) * Math.min(openSpace, 10) / 10.0; // Scale
																															// based
																															// on
																															// total
																															// open
																															// space
			}
		}

		// Penalize being too close to edges
		if (row < MIN_SAFE_ROW) {
			score -= 50 * (MIN_SAFE_ROW - row); // Progressive penalty
		} else if (row > MAX_SAFE_ROW) {
			score -= 50 * (row - MAX_SAFE_ROW); // Progressive penalty
		}

		return score;
	}

	/**
	 * Updates internal performance metrics to allow the controller to adjust its
	 * behavior over time.
	 * 
	 * @param model     The game grid
	 * @param playerRow The current player row
	 * @param action    The action taken
	 */
	private void updatePerformanceMetrics(LinkedList<byte[]> model, int playerRow, int action) {
		// Check if this was a successful move (no immediate danger)
		int newRow = playerRow + action;
		if (newRow >= 0 && newRow < MODEL_HEIGHT) {
			int nextColumnIndex = (PLAYER_COLUMN + 1) % model.size();
			boolean success = model.get(nextColumnIndex)[newRow] == 0;

			if (success) {
				successfulMoves++;
				recentPerformance[performanceIndex] = 1.0;
			} else {
				recentPerformance[performanceIndex] = 0.0;
			}

			// Update index
			performanceIndex = (performanceIndex + 1) % recentPerformance.length;

			// Adapt confidence threshold based on recent performance
			if (successfulMoves % 10 == 0) {
				adaptConfidenceThreshold();
			}
		}
	}

	/**
	 * Adapts the confidence threshold based on recent performance. If performing
	 * well, increase threshold to rely more on neural network. If performing
	 * poorly, decrease threshold to rely more on fuzzy controller.
	 */
	private void adaptConfidenceThreshold() {
		// Calculate average recent performance
		double sum = 0;
		for (double perf : recentPerformance) {
			sum += perf;
		}
		double avgPerformance = sum / recentPerformance.length;

		// Adjust confidence threshold based on performance
		if (avgPerformance > 0.95) {
			// Performing very well, trust NN more
			confidenceThreshold = Math.min(0.7, confidenceThreshold + 0.05);
		} else if (avgPerformance < 0.8) {
			// Performing poorly, trust fuzzy more
			confidenceThreshold = Math.max(0.3, confidenceThreshold - 0.05);
		}

		GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FUZZY_CONTROLLER, String.format(
				"Adapted confidence threshold to %.2f (performance: %.2f)", confidenceThreshold, avgPerformance));
	}

	/**
	 * Gets the last calculated tunnel centers.
	 * 
	 * @return Array of tunnel center positions
	 */
	public int[] getLastTunnelCenters() {
		return lastTunnelCenters != null ? lastTunnelCenters : new int[LOOK_AHEAD];
	}

	/**
	 * Gets the last calculated target row.
	 * 
	 * @return The target row
	 */
	public int getLastTargetRow() {
		return lastTargetRow;
	}

	/**
	 * Gets the last danger state.
	 * 
	 * @return true if the last state was dangerous
	 */
	public boolean getLastDangerState() {
		return lastDangerState;
	}

	/**
	 * Get performance metrics as a formatted string.
	 * 
	 * @return Performance metrics summary
	 */
	public String getPerformanceMetrics() {
		double overrideRate = successfulMoves + overriddenMoves > 0
				? (double) overriddenMoves / (successfulMoves + overriddenMoves)
				: 0;
		return String.format("Success: %d, Overrides: %d (%.1f%%), Threshold: %.2f", successfulMoves, overriddenMoves,
				overrideRate * 100, confidenceThreshold);
	}

	/**
	 * Returns the current confidence threshold.
	 * 
	 * @return The current confidence threshold
	 */
	public double getConfidenceThreshold() {
		return confidenceThreshold;
	}
}