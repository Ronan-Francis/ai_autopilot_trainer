package ie.atu.sw.game;

import java.util.LinkedList;
import java.util.List;

import ie.atu.sw.network.AutopilotSystem;
import ie.atu.sw.network.FlightData;
import ie.atu.sw.network.GameLogger;
import ie.atu.sw.network.GameLogger.LogCategory;
import ie.atu.sw.network.GameLogger.LogLevel;
import ie.atu.sw.ui.UIManager;

/**
 * Simplified AutopilotManager that acts as a facade to the AutopilotSystem.
 * Updated with training progression functionality.
 */
public class AutopilotManager {
    // Singleton instance
    private static AutopilotManager instance;
    
    // Core autopilot system
    private AutopilotSystem autopilotSystem;
    
    // Custom UiManager
    private UIManager uiManager;
    
    // Initialization flag
    private boolean isInitialized = false;
    
    /**
     * Private constructor to enforce singleton pattern.
     * No longer initializes UIManager to avoid circular dependencies.
     */
    private AutopilotManager() {
        GameLogger.log(LogLevel.INFO, LogCategory.GENERAL, "AutopilotManager initialized");
        // No longer initializing UIManager here
    }
    
    public long getCurrentFlightTime() {
        if (autopilotSystem != null) {
            return autopilotSystem.getCurrentFlightTime();
        }
        return 0;
    }
    
    public FlightData getCurrentFlightData() {
        if (autopilotSystem != null) {
            return autopilotSystem.getCurrentFlightData();
        }
        return null;
    }

    
    /**
     * Gets the singleton instance of AutopilotManager.
     */
    public static synchronized AutopilotManager getInstance() {
        if (instance == null) {
            instance = new AutopilotManager();
        }
        return instance;
    }

    /**
     * New method to complete initialization after instance creation.
     * This separates object creation from initialization to avoid circular dependencies.
     */
    public void completeInitialization() {
        if (!isInitialized) {
            // Get UIManager instance
            this.uiManager = UIManager.getInstance();
            
            // Initialize components
            initializeComponents();
            
            isInitialized = true;
            GameLogger.log(LogLevel.INFO, LogCategory.GENERAL, "AutopilotManager fully initialized");
        }
    }
    
    /**
     * Initialize components for the autopilot.
     */
    public void initializeComponents() {
        // Create the autopilot system with progressive training capabilities
        this.autopilotSystem = new AutopilotSystem();
        
        GameLogger.log(LogLevel.INFO, LogCategory.GENERAL, "Autopilot components initialized");
    }
    
    /**
     * Provides access to the AutopilotSystem for UI components.
     * This allows UI elements to directly query the system for training status.
     * 
     * @return The AutopilotSystem instance
     */
    public AutopilotSystem getAutopilotSystem() {
        return autopilotSystem;
    }
    
    /**
     * Determines the action to take based on the current state.
     */
    public int determineAction(LinkedList<byte[]> model, int playerRow) {
        if (autopilotSystem == null) {
            GameLogger.log(LogLevel.WARNING, LogCategory.NEURAL_NETWORK, 
                          "Autopilot system not initialized");
            return 0; // Default to stay
        }
        
        // Update the UI with current game state
        if (uiManager != null) {
            uiManager.updateGameState(model, playerRow);
        }
        
        return autopilotSystem.determineAction(model, playerRow);
    }

    /**
     * Records a move for training purposes.
     */
    public void recordMove(LinkedList<byte[]> model, int playerRow, int step) {
        // This is now handled directly by the autopilot system during determineAction
    }

    /**
     * Starts recording flight metrics.
     */
    public void startRecording() {
        if (autopilotSystem != null) {
            autopilotSystem.startRecording();
            if (uiManager != null) {
                uiManager.notifyRecordingStarted();
            }
        }
    }
    
    /**
     * Stops recording flight metrics.
     */
    public void stopRecording(boolean crashed) {
        if (autopilotSystem != null) {
            autopilotSystem.stopRecording(crashed);
            if (uiManager != null) {
                uiManager.notifyRecordingStopped();
                
                if (crashed) {
                    uiManager.notifyCrash();
                } else {
                    uiManager.notifyFlightCompleted();
                }
            }
        }
    }
    
    /**
     * Updates the current flight time.
     */
    public void updateFlightTime(long time) {
        if (autopilotSystem != null) {
            autopilotSystem.updateFlightTime(time);
        }
    }

    /**
     * Clears training data.
     */
    public void clearTrainingData() {
        if (autopilotSystem != null) {
            autopilotSystem.clearTrainingData();
        }
    }
    
    /**
     * Manually retrain the neural network model.
     */
    public boolean retrainModel() {
        if (autopilotSystem != null) {
            if (uiManager != null) {
                uiManager.notifyTrainingStarted();
            }
            boolean result = autopilotSystem.retrainModel();
            return result;
        }
        return false;
    }
    
    /**
     * Cycles the autopilot mode.
     */
    public AutopilotSystem.Mode cycleMode() {
        if (autopilotSystem != null) {
            AutopilotSystem.Mode newMode = autopilotSystem.cycleMode();
            if (uiManager != null) {
                uiManager.notifyModeChange();
            }
            return newMode;
        }
        return null;
    }
    
    /**
     * Gets the current autopilot mode.
     */
    public AutopilotSystem.Mode getMode() {
        if (autopilotSystem != null) {
            return autopilotSystem.getCurrentMode();
        }
        return null;
    }
    
    /**
     * Checks if recording is active.
     */
    public boolean isRecording() {
        return autopilotSystem != null && autopilotSystem.isRecording();
    }
    
    /**
     * Checks if retraining is in progress.
     */
    public boolean isRetraining() {
        return autopilotSystem != null && autopilotSystem.isTraining();
    }
    
    /**
     * Toggles the visibility of the autopilot performance monitor.
     */
    public void toggleMonitorVisibility() {
        if (uiManager != null) {
            uiManager.toggleAutopilotMonitor();
        }
    }
    
    public void toggleTeacherPanel() {
        if (uiManager != null) {
            uiManager.toggleTeacherPanel();
        }
    }
    
    public void toggleFlightAnalyzer() {
        if (uiManager != null) {
            uiManager.toggleFlightAnalyzer();
        }
    }
    
    public void shutdown() {
        GameLogger.log(LogLevel.INFO, LogCategory.GENERAL, "AutopilotManager shutting down");
        
        // Stop the autopilot system if necessary
        if (autopilotSystem != null && isRecording()) {
            stopRecording(false);
        }
        
        // Shutdown the UI manager
        if (uiManager != null) {
            uiManager.shutdown();
        }
    }

    public List<FlightData> getFlightHistory() {
        return autopilotSystem != null ? autopilotSystem.getFlightHistory() : null;
    }
}