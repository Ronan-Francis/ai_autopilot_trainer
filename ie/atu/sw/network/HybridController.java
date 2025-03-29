package ie.atu.sw.network;

import java.util.LinkedList;

/**
 * A hybrid controller that combines the optimized neural network with
 * a more conservative fuzzy controller for safety.
 */
public class HybridController {
	private static final int MODEL_HEIGHT = 20;
	private static final int PLAYER_COLUMN = 15;
    private NeuralNetworkController nnController;
    private FuzzyController fuzzyController;
    private boolean useHybridMode = true;
    
    /**
     * Creates a new hybrid controller.
     * 
     * @param nnController The optimized neural network controller
     * @param fuzzyController The improved fuzzy controller
     */
    public HybridController(NeuralNetworkController nnController, FuzzyController fuzzyController) {
        this.nnController = nnController;
        this.fuzzyController = fuzzyController;
    }
    
    /**
     * Determines the best action to take using both controllers with improved
     * disagreement resolution and geometry analysis.
     * 
     * @param model The game grid
     * @param playerRow The current player row
     * @return The action to take (-1=up, 0=stay, 1=down)
     */
    public int determineAction(LinkedList<byte[]> model, int playerRow) {
        // Get neural network prediction
        int nnAction = nnController.predictAction(model, playerRow);
        double[] confidenceScores = nnController.getLastOutputScores();
        
        // Calculate highest confidence score
        double highestConfidence = 0;
        for (double score : confidenceScores) {
            if (score > highestConfidence) {
                highestConfidence = score;
            }
        }
        
        int action = nnAction;
        
        // Check if we should use hybrid mode
        if (useHybridMode) {
            // Get fuzzy controller recommendation
            int fuzzyAction = fuzzyController.determineAction(model, playerRow);
            
            // Safety check: analyze the immediate tunnel geometry
            // Look at next column to determine if moving up or down is safer
            int nextColumnIndex = (PLAYER_COLUMN + 1) % model.size();
            byte[] nextColumn = model.get(nextColumnIndex);
            
            // Find tunnel top and bottom in next column
            int topEdge = 0;
            while (topEdge < MODEL_HEIGHT && nextColumn[topEdge] != 0) {
                topEdge++;
            }
            
            int bottomEdge = MODEL_HEIGHT - 1;
            while (bottomEdge >= 0 && nextColumn[bottomEdge] != 0) {
                bottomEdge--;
            }
            
            // Check if tunnel is valid
            if (topEdge <= bottomEdge) {
                int tunnelCenter = (topEdge + bottomEdge) / 2;
                int tunnelHeight = bottomEdge - topEdge + 1;
                
                // Calculate space above and below player
                int spaceAbove = playerRow - topEdge;
                int spaceBelow = bottomEdge - playerRow;
                
                // Log geometry analysis
                GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                    "Geometry analysis - Center: " + tunnelCenter + ", PlayerRow: " + playerRow + 
                    ", Space above: " + spaceAbove + ", Space below: " + spaceBelow);
                
                // Check if fuzzy controller should override
                Integer overrideAction = fuzzyController.checkForOverride(
                    model, playerRow, nnAction, highestConfidence, confidenceScores);
                    
                if (overrideAction != null) {
                    // Use fuzzy controller override
                    action = overrideAction;
                    
                    // Log the override
                    GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                        String.format("Fuzzy override: NN action %s -> Fuzzy action %s", 
                                      actionToString(nnAction), actionToString(action)));
                }
                // When decisions disagree, use geometry analysis to decide
                else if (fuzzyAction != nnAction) {
                    boolean safer = false;
                    
                    // If neural network wants to move UP
                    if (nnAction == -1) {
                        // Check if moving up is actually safer
                        if (spaceAbove > spaceBelow || tunnelCenter < playerRow) {
                            safer = true;
                        }
                    }
                    // If neural network wants to move DOWN
                    else if (nnAction == 1) {
                        // Check if moving down is actually safer
                        if (spaceBelow > spaceAbove || tunnelCenter > playerRow) {
                            safer = true;
                        }
                    }
                    
                    if (safer) {
                        // Neural network action seems safer based on geometry
                        action = nnAction;
                        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                            "Geometry confirms NN action is safer: " + actionToString(action));
                    } else {
                        // Fuzzy action seems safer based on geometry
                        action = fuzzyAction;
                        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                            "Geometry suggests Fuzzy action is safer: " + actionToString(action));
                    }
                }
                
                // CRITICAL SAFETY FEATURE: Always move toward more open space if in danger
                boolean inDanger = fuzzyController.getLastDangerState();
                if (inDanger) {
                    // Find which direction has more space
                    int newDirection;
                    if (spaceAbove > spaceBelow + 2) { // Significant more space above
                        newDirection = -1; // Move UP
                    } else if (spaceBelow > spaceAbove + 2) { // Significantly more space below
                        newDirection = 1; // Move DOWN
                    } else if (Math.abs(tunnelCenter - playerRow) > 2) {
                        // Move toward tunnel center when significantly off-center
                        newDirection = (tunnelCenter < playerRow) ? -1 : 1;
                    } else {
                        // Keep current direction if no clear advantage
                        newDirection = action;
                    }
                    
                    // If we're changing the action for safety, log it
                    if (newDirection != action) {
                        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                            String.format("SAFETY OVERRIDE: Action %s -> %s (more space)",
                                         actionToString(action), actionToString(newDirection)));
                        action = newDirection;
                    }
                }
            }
        }
        
        // Log final decision
        GameLogger.logFinalAction(action);
        
        return action;
    }
    
    
    /**
     * Toggles hybrid mode on/off.
     */
    public void toggleHybridMode() {
        useHybridMode = !useHybridMode;
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
            "Hybrid mode " + (useHybridMode ? "enabled" : "disabled"));
    }
    
    /**
     * Checks if hybrid mode is enabled.
     * 
     * @return true if hybrid mode is enabled
     */
    public boolean isHybridModeEnabled() {
        return useHybridMode;
    }
    
    /**
     * Gets the fuzzy controller's performance metrics.
     * 
     * @return The performance metrics string
     */
    public String getFuzzyPerformanceMetrics() {
        return fuzzyController.getPerformanceMetrics();
    }
    
    /**
     * Gets the fuzzy controller's confidence threshold.
     * 
     * @return The confidence threshold
     */
    public double getFuzzyConfidenceThreshold() {
        return fuzzyController.getConfidenceThreshold();
    }
    
    /**
     * Converts an action integer to a human-readable string.
     * 
     * @param action The action integer (-1=up, 0=stay, 1=down)
     * @return A human-readable string representation
     */
    private String actionToString(int action) {
        return switch (action) {
            case -1 -> "UP";
            case 0 -> "STAY";
            case 1 -> "DOWN";
            default -> "UNKNOWN";
        };
    }
}