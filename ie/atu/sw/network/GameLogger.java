package ie.atu.sw.network;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedList;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * A central logging system for the game that handles different types of logs: -
 * General application logs - Model decision logs (neural network and fuzzy
 * controller) - Performance metrics
 */
public class GameLogger {
	// Log levels
	public enum LogLevel {
		DEBUG, INFO, WARNING, ERROR
	}

	// Log categories
	public enum LogCategory {
		GENERAL, NEURAL_NETWORK, FUZZY_CONTROLLER, FEATURE_EXTRACTOR, GAME_STATE
	}

	// Constants
	private static final String LOG_DIRECTORY = "./logs/";
	private static final String GENERAL_LOG_FILE = "game.log";
	private static final String DECISIONS_LOG_FILE = "decisions.log";
	private static final boolean CONSOLE_OUTPUT = true;

	// Queue for log messages (thread-safe)
	private static final ConcurrentLinkedQueue<String> logQueue = new ConcurrentLinkedQueue<>();

	// Writer threads
	private static LogWriterThread generalLogWriter;
	private static LogWriterThread decisionsLogWriter;

	// Format for timestamps
	private static final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");

	// Game session identifier
	private static final String sessionId = generateSessionId();

	// Latest decision data for UI display
	private static String lastNeuralNetworkDecision = "N/A";
	private static String lastFuzzyControllerDecision = "N/A";
	private static String lastActualDecision = "N/A";
	private static double[] lastConfidenceScores = null;

	/**
	 * Initializes the logging system.
	 */
	public static void initialize() {
		try {
			// Create log directory if it doesn't exist
			File logDir = new File(LOG_DIRECTORY);
			if (!logDir.exists()) {
				logDir.mkdirs();
			}

			// Start log writer threads
			generalLogWriter = new LogWriterThread(LOG_DIRECTORY + GENERAL_LOG_FILE);
			decisionsLogWriter = new LogWriterThread(LOG_DIRECTORY + DECISIONS_LOG_FILE);

			generalLogWriter.start();
			decisionsLogWriter.start();

			// Log that the logging system has been initialized
			log(LogLevel.INFO, LogCategory.GENERAL, "Logging system initialized. Session ID: " + sessionId);
		} catch (Exception e) {
			System.err.println("Failed to initialize logging system: " + e.getMessage());
			e.printStackTrace();
		}
	}

	/**
	 * Logs a message with the specified level and category.
	 * 
	 * @param level    The log level
	 * @param category The log category
	 * @param message  The log message
	 */
	public static void log(LogLevel level, LogCategory category, String message) {
		String timestamp = dateFormat.format(new Date());
		String logMessage = String.format("[%s] [%s] [%s] [%s] %s", timestamp, sessionId, level, category, message);

		// Add to queue for writing to file
		logQueue.add(logMessage);

		// Print to console if enabled
		if (CONSOLE_OUTPUT) {
			System.out.println(logMessage);
		}
	}

	/**
	 * Logs a decision made by the neural network.
	 * 
	 * @param features        The input features
	 * @param outputScores    The output scores (confidence)
	 * @param predictedAction The action predicted by the neural network
	 * @param actualAction    The action actually taken
	 * @param playerRow       The current player row
	 * @param override        Whether the decision was overridden by the fuzzy
	 *                        controller
	 */
	public static void logNeuralNetworkDecision(double[] features, double[] outputScores, int predictedAction,
			int actualAction, int playerRow, boolean override, String decider) {
		// Convert to human-readable action names
		String predictedActionName = actionToString(predictedAction);
		String actualActionName = actionToString(actualAction);

		// Format the feature vector
		StringBuilder featureStr = new StringBuilder("[");
		for (int i = 0; i < features.length; i++) {
			featureStr.append(String.format("%.3f", features[i]));
			if (i < features.length - 1)
				featureStr.append(", ");
		}
		featureStr.append("]");

		// Format the output scores
		StringBuilder scoresStr = new StringBuilder("[");
		for (int i = 0; i < outputScores.length; i++) {
			scoresStr.append(String.format("%.3f", outputScores[i]));
			if (i < outputScores.length - 1)
				scoresStr.append(", ");
		}
		scoresStr.append("]");

		// Create decision log message
		String decisionLog = String.format(
				"NN Decision: %s (%.2f%%), Actual: %s, Player Row: %d, Features: %s, Scores: %s%s", predictedActionName,
				outputScores[predictedAction + 1] * 100, // +1 to adjust for action mapping
				actualActionName, playerRow, featureStr.toString(), scoresStr.toString(),
				override ? " [OVERRIDDEN BY FUZZY]" : "");

		// Store for UI display
		lastNeuralNetworkDecision = predictedActionName
				+ String.format(" (%.1f%%)", outputScores[predictedAction + 1] * 100);
		lastConfidenceScores = outputScores;

		// Write to decisions log
		decisionsLogWriter.queueMessage(decisionLog);

		// Also log a shorter version to the general log
		log(LogLevel.INFO, LogCategory.NEURAL_NETWORK,
				String.format("NN: %s -> %s, Confidence: %.2f%%%s", predictedActionName, actualActionName,
						outputScores[predictedAction + 1] * 100, override ? " [OVERRIDDEN] by " + decider : ""));
	}

	/**
	 * Logs a decision made by the fuzzy controller.
	 * 
	 * @param playerRow     The current player row
	 * @param tunnelCenters The calculated tunnel centers
	 * @param targetRow     The target row calculated by the fuzzy controller
	 * @param action        The action decided by the fuzzy controller
	 * @param danger        Whether the controller detected immediate danger
	 */
	public static void logFuzzyControllerDecision(int playerRow, int[] tunnelCenters, int targetRow, int action,
			boolean danger) {
		// Convert to human-readable action name
		String actionName = actionToString(action);

		// Format the tunnel centers
		StringBuilder centersStr = new StringBuilder("[");
		for (int i = 0; i < tunnelCenters.length; i++) {
			centersStr.append(tunnelCenters[i]);
			if (i < tunnelCenters.length - 1)
				centersStr.append(", ");
		}
		centersStr.append("]");

		// Create decision log message
		String decisionLog = String.format(
				"Fuzzy Decision: %s, Player Row: %d, Target: %d, Tunnel Centers: %s, Danger: %s", actionName, playerRow,
				targetRow, centersStr.toString(), danger ? "YES" : "NO");

		// Store for UI display
		lastFuzzyControllerDecision = actionName + (danger ? " (DANGER)" : "");

		// Write to decisions log
		decisionsLogWriter.queueMessage(decisionLog);

		// Also log a shorter version to the general log
		log(LogLevel.INFO, LogCategory.FUZZY_CONTROLLER, String.format("Fuzzy: %s, Player: %d, Target: %d, Danger: %s",
				actionName, playerRow, targetRow, danger ? "YES" : "NO"));
	}

	/**
	 * Logs the final action taken after all decision systems have had input.
	 * 
	 * @param action The final action taken
	 */
	public static void logFinalAction(int action) {
		String actionName = actionToString(action);
		lastActualDecision = actionName;

		log(LogLevel.INFO, LogCategory.GAME_STATE, "Final action: " + actionName);
	}

	/**
	 * Logs the features extracted from the current game state.
	 * 
	 * @param features  The feature vector
	 * @param model     The game model
	 * @param playerRow The player row
	 */
	public static void logFeatureExtraction(double[] features, LinkedList<byte[]> model, int playerRow) {
		StringBuilder sb = new StringBuilder("Extracted features: [");
		for (int i = 0; i < features.length; i++) {
			sb.append(String.format("%.3f", features[i]));
			if (i < features.length - 1)
				sb.append(", ");
		}
		sb.append("]");

		log(LogLevel.DEBUG, LogCategory.FEATURE_EXTRACTOR, sb.toString());
	}

	/**
	 * Logs game state information.
	 * 
	 * @param playerRow The current player row
	 * @param time      The current game time
	 */
	public static void logGameState(int playerRow, long time) {
		log(LogLevel.DEBUG, LogCategory.GAME_STATE, String.format("Player Row: %d, Time: %d", playerRow, time));
	}

	/**
	 * Logs an exception.
	 * 
	 * @param category The log category
	 * @param message  A description of when/where the exception occurred
	 * @param e        The exception
	 */
	public static void logException(LogCategory category, String message, Exception e) {
		StringWriter sw = new StringWriter();
		e.printStackTrace(new PrintWriter(sw));

		log(LogLevel.ERROR, category, message + ": " + e.getMessage());
		log(LogLevel.ERROR, category, "Stack trace: " + sw.toString());
	}

	/**
	 * Converts an action integer to a human-readable string.
	 * 
	 * @param action The action integer (-1=up, 0=stay, 1=down)
	 * @return A human-readable string representation
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
	 * Generates a unique session ID for this game instance.
	 * 
	 * @return A unique session ID
	 */
	private static String generateSessionId() {
		SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd-HHmmss");
		return sdf.format(new Date()) + "-" + System.nanoTime() % 10000;
	}

	/**
	 * Shutdown hook to properly close log files.
	 */
	public static void shutdown() {
		log(LogLevel.INFO, LogCategory.GENERAL, "Shutting down logging system");

		if (generalLogWriter != null) {
			generalLogWriter.shutdown();
		}

		if (decisionsLogWriter != null) {
			decisionsLogWriter.shutdown();
		}
	}

	/**
	 * Returns the latest neural network decision for UI display.
	 * 
	 * @return The latest neural network decision
	 */
	public static String getLastNeuralNetworkDecision() {
		return lastNeuralNetworkDecision;
	}

	/**
	 * Returns the latest fuzzy controller decision for UI display.
	 * 
	 * @return The latest fuzzy controller decision
	 */
	public static String getLastFuzzyControllerDecision() {
		return lastFuzzyControllerDecision;
	}

	/**
	 * Returns the latest actual decision for UI display.
	 * 
	 * @return The latest actual decision
	 */
	public static String getLastActualDecision() {
		return lastActualDecision;
	}

	/**
	 * Returns the latest confidence scores for UI display.
	 * 
	 * @return The latest confidence scores
	 */
	public static double[] getLastConfidenceScores() {
		return lastConfidenceScores;
	}

	/**
	 * A thread that writes log messages to a file.
	 */
	private static class LogWriterThread extends Thread {
		private final String filePath;
		private final ConcurrentLinkedQueue<String> messageQueue = new ConcurrentLinkedQueue<>();
		private volatile boolean running = true;

		public LogWriterThread(String filePath) {
			this.filePath = filePath;
			setDaemon(true); // Make this a daemon thread
		}

		public void queueMessage(String message) {
			messageQueue.add(message);
		}

		@Override
		public void run() {
			try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath, true))) {
				// Write header
				writer.write("=== NEW SESSION STARTED: " + sessionId + " ===\n");
				writer.flush();

				while (running || !messageQueue.isEmpty()) {
					String message = messageQueue.poll();

					if (message != null) {
						writer.write(message + "\n");
						writer.flush();
					} else {
						// No messages in queue, sleep for a bit
						try {
							Thread.sleep(100);
						} catch (InterruptedException e) {
							// Ignore
						}
					}
				}

				// Write footer
				writer.write("=== SESSION ENDED: " + sessionId + " ===\n");
				writer.flush();
			} catch (IOException e) {
				System.err.println("Error writing to log file: " + e.getMessage());
			}
		}

		public void shutdown() {
			running = false;

			try {
				join(1000); // Wait for up to 1 second for the thread to finish
			} catch (InterruptedException e) {
				// Ignore
			}
		}
	}

	/**
	 * Enhanced version of the fuzzy controller decision logging method that
	 * includes additional information about tunnel trends and widths.
	 * 
	 * @param playerRow     The current player row
	 * @param tunnelCenters The calculated tunnel centers
	 * @param tunnelWidths  The calculated tunnel widths (optional)
	 * @param targetRow     The target row calculated by the fuzzy controller
	 * @param action        The action decided by the fuzzy controller
	 * @param danger        Whether the controller detected immediate danger
	 * @param trend         The tunnel movement trend (optional)
	 */
	public static void logEnhancedFuzzyDecision(int playerRow, int[] tunnelCenters, double[] tunnelWidths,
			int targetRow, int action, boolean danger, String trend) {
		// Convert to human-readable action name
		String actionName = actionToString(action);

		// Format the tunnel centers
		StringBuilder centersStr = new StringBuilder("[");
		for (int i = 0; i < tunnelCenters.length; i++) {
			centersStr.append(tunnelCenters[i]);
			if (i < tunnelCenters.length - 1)
				centersStr.append(", ");
		}
		centersStr.append("]");

		// Format the tunnel widths if provided
		StringBuilder widthsStr = new StringBuilder("[");
		if (tunnelWidths != null) {
			for (int i = 0; i < tunnelWidths.length; i++) {
				widthsStr.append(String.format("%.1f", tunnelWidths[i]));
				if (i < tunnelWidths.length - 1)
					widthsStr.append(", ");
			}
		}
		widthsStr.append("]");

		// Create enhanced decision log message
		String decisionLog = String.format(
				"Enhanced Fuzzy Decision: %s, Player Row: %d, Target: %d, Tunnel Centers: %s, Widths: %s, Trend: %s, Danger: %s",
				actionName, playerRow, targetRow, centersStr.toString(), widthsStr.toString(), trend,
				danger ? "YES" : "NO");

		// Store for UI display
		lastFuzzyControllerDecision = actionName + (danger ? " (DANGER)" : "") + " [" + trend + "]";

		// Write to decisions log
		decisionsLogWriter.queueMessage(decisionLog);

		// Also log a shorter version to the general log
		log(LogLevel.INFO, LogCategory.FUZZY_CONTROLLER,
				String.format("Fuzzy: %s, Player: %d, Target: %d, Trend: %s, Danger: %s", actionName, playerRow,
						targetRow, trend, danger ? "YES" : "NO"));
	}
}