package ie.atu.sw.network;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Simplified logging system for the game.
 */
public class GameLogger {
    // Log levels
    public enum LogLevel {
        DEBUG, INFO, WARNING, ERROR
    }
    
    // Log categories
    public enum LogCategory {
        GENERAL, NEURAL_NETWORK, FEATURE_EXTRACTOR, GAME_STATE
    }
    
    // Constants
    private static final String LOG_FILE = "./logs/game.log";
    private static final boolean CONSOLE_OUTPUT = true;
    
    // Latest decision data for UI display
    private static String lastNeuralNetworkDecision = "N/A";
    private static String lastActualDecision = "N/A";
    private static double[] lastConfidenceScores = null;
    
    // Format for timestamps
    private static final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
    
    // Thread pool for asynchronous logging
    private static final ExecutorService logExecutor = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "GameLogger-Thread");
        t.setDaemon(true);
        return t;
    });
    
    // Initialization - ensure log directory exists
    static {
        try {
            File logDir = new File("./logs");
            if (!logDir.exists()) {
                logDir.mkdirs();
            }
        } catch (Exception e) {
            System.err.println("Failed to initialize log directory: " + e.getMessage());
        }
    }
    
    /**
     * Logs a message with the specified level and category.
     */
    public static void log(LogLevel level, LogCategory category, String message) {
        String timestamp = dateFormat.format(new Date());
        String logMessage = String.format("[%s] [%s] [%s] %s", timestamp, level, category, message);
        
        // Print to console if enabled
        if (CONSOLE_OUTPUT) {
            System.out.println(logMessage);
        }
        
        // Write to log file asynchronously
        logExecutor.submit(() -> {
            try (FileWriter writer = new FileWriter(LOG_FILE, true)) {
                writer.write(logMessage + "\n");
            } catch (Exception e) {
                System.err.println("Error writing to log file: " + e.getMessage());
            }
        });
    }
    
    /**
     * Logs an exception.
     */
    public static void logException(LogCategory category, String message, Exception e) {
        StringWriter sw = new StringWriter();
        e.printStackTrace(new PrintWriter(sw));
        
        log(LogLevel.ERROR, category, message + ": " + e.getMessage());
        log(LogLevel.ERROR, category, "Stack trace: " + sw.toString());
    }
    
    /**
     * Logs a neural network decision.
     */
    public static void logNeuralNetworkDecision(double[] features, double[] outputScores, 
                                               int predictedAction, int actualAction, int playerRow) {
        // Convert to human-readable action names
        String predictedActionName = actionToString(predictedAction);
        String actualActionName = actionToString(actualAction);
        
        // Create decision log message
        String decisionLog = String.format(
                "NN Decision: %s (%.2f%%), Actual: %s, Player Row: %d", 
                predictedActionName,
                outputScores[predictedAction + 1] * 100, // +1 to adjust for action mapping
                actualActionName, playerRow);
        
        // Store for UI display
        lastNeuralNetworkDecision = predictedActionName
                + String.format(" (%.1f%%)", outputScores[predictedAction + 1] * 100);
        lastConfidenceScores = outputScores;
        lastActualDecision = actualActionName;
        
        // Log to general log
        log(LogLevel.INFO, LogCategory.NEURAL_NETWORK, decisionLog);
    }
    
    /**
     * Logs the final action taken.
     */
    public static void logFinalAction(int action) {
        String actionName = actionToString(action);
        lastActualDecision = actionName;
        
        log(LogLevel.INFO, LogCategory.GAME_STATE, "Final action: " + actionName);
    }
    
    /**
     * Converts an action integer to a human-readable string.
     */
    private static String actionToString(int action) {
        return switch (action) {
            case -1 -> "UP";
            case 0 -> "STAY";
            case 1 -> "DOWN";
            default -> "UNKNOWN";
        };
    }
    
    /**
     * Returns the latest neural network decision for UI display.
     */
    public static String getLastNeuralNetworkDecision() {
        return lastNeuralNetworkDecision;
    }
    
    /**
     * Returns the latest actual decision for UI display.
     */
    public static String getLastActualDecision() {
        return lastActualDecision;
    }
    
    /**
     * Returns the latest confidence scores for UI display.
     */
    public static double[] getLastConfidenceScores() {
        return lastConfidenceScores;
    }
    
    /**
     * Shuts down the logger, ensuring all pending log messages are written.
     * This method should be called when the application is shutting down.
     * 
     * @param timeout The maximum time to wait for pending logs to be written, in seconds
     * @return true if shutdown completed normally, false if timeout occurred
     */
    public static boolean shutdown(int timeout) {
        try {
            log(LogLevel.INFO, LogCategory.GENERAL, "GameLogger shutting down...");
            
            // Reject new tasks and begin shutdown
            logExecutor.shutdown();
            
            // Wait for existing tasks to terminate
            boolean terminated = logExecutor.awaitTermination(timeout, TimeUnit.SECONDS);
            
            if (!terminated) {
                // Force shutdown if timeout occurs
                logExecutor.shutdownNow();
                System.err.println("GameLogger shutdown timed out after " + timeout + " seconds");
                return false;
            }
            
            return true;
        } catch (InterruptedException e) {
            // Restore the interrupted status
            Thread.currentThread().interrupt();
            System.err.println("GameLogger shutdown interrupted: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Shuts down the logger with a default timeout of 5 seconds.
     * 
     * @return true if shutdown completed normally, false if timeout occurred
     */
    public static boolean shutdown() {
        return shutdown(5);
    }

}