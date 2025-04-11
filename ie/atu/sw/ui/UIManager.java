package ie.atu.sw.ui;

import java.util.LinkedList;

import ie.atu.sw.network.GameLogger;

/**
 * Central manager for all UI components, providing a unified interface
 * for the game to interact with the UI.
 */
public class UIManager {
    // Singleton instance
    private static UIManager instance;
    
    // UI components
    private AutopilotMonitor autopilotMonitor;
    private TeacherPanel teacherPanel;
    private FlightAnalyzer flightAnalyzer;
    
    // Initialization flag
    private boolean uiComponentsInitialized = false;
    
    /**
     * Gets the singleton instance of UIManager.
     */
    public static synchronized UIManager getInstance() {
        if (instance == null) {
            instance = new UIManager();
        }
        return instance;
    }
    
    /**
     * Private constructor to enforce singleton pattern.
     */
    private UIManager() {
        // Don't initialize UI components here to avoid circular dependencies
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                       "UI Manager instance created");
    }
    
    /**
     * Initializes all UI components.
     * This method must be called after AutopilotManager is fully constructed.
     */
    public void initialize() {
        if (!uiComponentsInitialized) {
            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                           "Initializing UI components");
                           
            // Set initialization flag first to prevent recursion
            uiComponentsInitialized = true;
            
            // Components will be lazily initialized when needed
            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                           "UI component initialization complete");
        }
    }
    
    /**
     * Gets or creates the AutopilotMonitor instance.
     */
    private AutopilotMonitor getAutopilotMonitor() {
        if (autopilotMonitor == null) {
            autopilotMonitor = AutopilotMonitor.getInstance();
        }
        return autopilotMonitor;
    }
    
    /**
     * Gets or creates the TeacherPanel instance.
     */
    private TeacherPanel getTeacherPanel() {
        if (teacherPanel == null) {
            teacherPanel = TeacherPanel.getInstance();
        }
        return teacherPanel;
    }
    
    /**
     * Gets or creates the FlightAnalyzer instance.
     */
    private FlightAnalyzer getFlightAnalyzer() {
        if (flightAnalyzer == null) {
            flightAnalyzer = FlightAnalyzer.getInstance();
        }
        return flightAnalyzer;
    }
    
    /**
     * Toggles the visibility of the autopilot monitor.
     */
    public void toggleAutopilotMonitor() {
        getAutopilotMonitor().toggleVisibility();
    }
    
    /**
     * Toggles the visibility of the teacher panel.
     */
    public void toggleTeacherPanel() {
        getTeacherPanel().toggleVisibility();
    }
    
    /**
     * Toggles the visibility of the flight analyzer.
     */
    public void toggleFlightAnalyzer() {
        getFlightAnalyzer().toggleVisibility();
    }
    
    /**
     * Updates the game state in the teacher panel.
     * 
     * @param model The current game model
     * @param playerRow The current player row
     */
    public void updateGameState(LinkedList<byte[]> model, int playerRow) {
        if (teacherPanel != null) {
            teacherPanel.updateGameState(model, playerRow);
        }
    }
    
    /**
     * Updates all real-time displays.
     */
    public void updateDisplays() {
        // Update any real-time displays here
    }
    
    /**
     * Notifies the UI of a mode change.
     */
    public void notifyModeChange() {
        // Update mode display if needed
    }
    
    /**
     * Notifies the UI that training has started.
     */
    public void notifyTrainingStarted() {
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
                       "UI notified: Training started");
    }
    
    /**
     * Notifies the UI that training has completed.
     */
    public void notifyTrainingCompleted() {
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
                       "UI notified: Training completed");
    }
    
    /**
     * Notifies the UI that recording has started.
     */
    public void notifyRecordingStarted() {
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                       "UI notified: Recording started");
    }
    
    /**
     * Notifies the UI that recording has stopped.
     */
    public void notifyRecordingStopped() {
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                       "UI notified: Recording stopped");
    }
    
    /**
     * Notifies the UI of a game crash.
     */
    public void notifyCrash() {
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                       "UI notified: Game crashed");
    }
    
    /**
     * Notifies the UI of a successful flight completion.
     */
    public void notifyFlightCompleted() {
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                       "UI notified: Flight completed successfully");
    }
    
    /**
     * Handles cleanup when the application is shutting down.
     */
    public void shutdown() {
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                       "UI Manager shutting down");
        
        // Perform any necessary cleanup
    }
}