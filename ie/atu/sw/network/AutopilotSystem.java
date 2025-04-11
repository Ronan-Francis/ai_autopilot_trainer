package ie.atu.sw.network;

import java.io.File;
import java.util.ArrayList;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.encog.ml.data.MLDataSet;

/**
 * Integrated autopilot system that combines feature extraction, decision
 * making, and flight recording, with progressive training capabilities.
 */
public class AutopilotSystem {
	// Autopilot modes
	public enum Mode {
		NEURAL_NETWORK, TEACHER, HYBRID
	}

	// Current mode
	private Mode currentMode;
	
	// Progressive training states
	private enum TrainingStage {
		TEACHER_ONLY, // Initial stage - using teacher only
		HYBRID_LEARNING, // Second stage - using both teacher and NN
		NEURAL_NETWORK // Final stage - using NN only
	}

	private TrainingStage currentTrainingStage = TrainingStage.TEACHER_ONLY;

	// Training progression thresholds
	private static final double CONFIDENCE_THRESHOLD = 0.6;
	private static final int CONFIDENCE_MATCH_COUNT_THRESHOLD = 25; // Number of decisions above threshold
	private static final int HYBRID_FLIGHT_SECONDS_THRESHOLD = 30; // Time in hybrid mode before switching to NN
	private static final int NN_FLIGHT_SECONDS_THRESHOLD = 600; // Time in NN mode before "graduation"

	// Training progression tracking
	private int confidenceMatchCount = 0;
	private int consecutiveHighConfidenceCount = 0;
	private boolean hybridModeAchieved = false;
	private boolean neuralNetworkModeAchieved = false;
	private long hybridModeStartTime = 0;
	private long neuralNetworkModeStartTime = 0;
	private boolean trainingCompleted = false;

	// Core components
	private NeuralNetworkSystem networkSystem;
	private SimplifiedFeatureExtractor featureExtractor;

	// Flight recording
	private boolean isRecording = false;
	private long flightStartTime;
	private long currentFlightTime = 0;

	// Current flight metrics
	private int totalActions = 0;
	private int upActions = 0;
	private int stayActions = 0;
	private int downActions = 0;
	private double totalConfidence = 0.0;
	private double minConfidence = 1.0;
	private final List<Double> confidenceHistory = new ArrayList<>();

	// Flight history
	private final List<FlightData> flightHistory = new ArrayList<>();
	private static final int MAX_HISTORY = 20;

	// Training
	private boolean isTraining = false;
	private ExecutorService trainingExecutor = Executors.newSingleThreadExecutor();

	// System constants
	private static final int MODEL_HEIGHT = 20;
	private static final int PLAYER_COLUMN = 15;

	/**
	 * Creates a new autopilot system.
	 */
	public AutopilotSystem() {
		// Create components
		this.featureExtractor = new SimplifiedFeatureExtractor();
		this.networkSystem = new NeuralNetworkSystem(featureExtractor.getFeatureCount(), 10); // Default hidden neurons

		// Determine initial mode based on model existence
		File modelFile = new File("resources/network_model.eg");
		if (modelFile.exists()) {
			this.currentMode = Mode.NEURAL_NETWORK;
		} else {
			this.currentMode = Mode.TEACHER;
		}

		// Reset progression tracking
		resetTrainingProgression();

		GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
				"Autopilot System initialized in progressive training mode");
	}

	/**
	 * Resets the training progression tracking.
	 */
	private void resetTrainingProgression() {
		confidenceMatchCount = 0;
		consecutiveHighConfidenceCount = 0;
		hybridModeAchieved = false;
		neuralNetworkModeAchieved = false;
		hybridModeStartTime = 0;
		neuralNetworkModeStartTime = 0;
		trainingCompleted = false;
	}

	/**
	 * Determines the action to take based on the current state.
	 */
	public int determineAction(LinkedList<byte[]> model, int playerRow) {
		// Extract features
		double[] features = featureExtractor.extractFeatures(model, playerRow);

		// Get network prediction (0=up, 1=stay, 2=down)
		int networkPrediction = networkSystem.predict(features);

		// Convert to game action (-1=up, 0=stay, 1=down)
		int networkAction = networkPrediction - 1;

		// Get teacher action
		int teacherAction = suggestTeacherAction(model, playerRow);

		// Get confidence scores for this prediction
		double[] confidenceScores = networkSystem.getLastOutputScores();
		double networkConfidence = confidenceScores[networkPrediction];

		// In progressive training mode, evaluate network vs teacher for progression
		if (!trainingCompleted && isRecording) {
			updateTrainingProgression(networkAction, teacherAction, networkConfidence);
		}

		// Determine final action based on mode
		int finalAction;
		switch (currentMode) {
		case NEURAL_NETWORK:
			finalAction = networkAction;
			break;
		case TEACHER:
			finalAction = teacherAction;
			break;
		case HYBRID:
			// Use teacher if network action would be dangerous
			if (isDangerousAction(model, playerRow, networkAction)) {
				finalAction = teacherAction;
				GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
						"Teacher overriding neural network action");
			} else {
				finalAction = networkAction;
			}
			break;
		default:
			finalAction = networkAction;
		}

		// Log decision
		GameLogger.logNeuralNetworkDecision(features, confidenceScores, networkAction, finalAction, playerRow);

		// If recording, track this action
		if (isRecording) {
			recordAction(finalAction, confidenceScores[networkPrediction]);
		}

		return finalAction;
	}

	/**
	 * Updates the training progression status based on current performance.
	 * 
	 * @param networkAction     The action suggested by the neural network
	 * @param teacherAction     The action suggested by the teacher
	 * @param networkConfidence The confidence of the neural network prediction
	 */
	private void updateTrainingProgression(int networkAction, int teacherAction, double networkConfidence) {
		// First check if the NN confidence is above threshold
		if (networkConfidence >= CONFIDENCE_THRESHOLD) {
			consecutiveHighConfidenceCount++;

			// If network matches teacher and has high confidence, increment match count
			if (networkAction == teacherAction) {
				confidenceMatchCount++;

				// Log milestone every 5 matches
				if (confidenceMatchCount % 5 == 0) {
					GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
							"Training progression: " + confidenceMatchCount + "/" + CONFIDENCE_MATCH_COUNT_THRESHOLD
									+ " high confidence matches");
				}
			}
		} else {
			// Reset consecutive high confidence count if we get a low confidence prediction
			consecutiveHighConfidenceCount = 0;
		}

		// Process training stage transitions based on current progression
		processTrainingStageTransitions();
	}

	/**
	 * Processes potential training stage transitions based on current performance.
	 */
	private void processTrainingStageTransitions() {
		switch (currentTrainingStage) {
		case TEACHER_ONLY:
			// Transition from TEACHER_ONLY to HYBRID_LEARNING if we've met the confidence
			// threshold
			if (!hybridModeAchieved && confidenceMatchCount >= CONFIDENCE_MATCH_COUNT_THRESHOLD && currentFlightTime >= 300) {
				hybridModeAchieved = true;
				currentTrainingStage = TrainingStage.HYBRID_LEARNING;
				setMode(Mode.HYBRID);
				hybridModeStartTime = currentFlightTime;

				GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
						"TRAINING PROGRESSION: Advanced to HYBRID mode after " + confidenceMatchCount
								+ " high confidence matches");
			}
			break;

		case HYBRID_LEARNING:
			// Check if we've been in hybrid mode long enough
			if (!neuralNetworkModeAchieved
					&& (currentFlightTime - hybridModeStartTime) >= HYBRID_FLIGHT_SECONDS_THRESHOLD) {
				neuralNetworkModeAchieved = true;
				currentTrainingStage = TrainingStage.NEURAL_NETWORK;
				setMode(Mode.NEURAL_NETWORK);
				neuralNetworkModeStartTime = currentFlightTime;

				GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
						"TRAINING PROGRESSION: Advanced to NEURAL NETWORK mode after "
								+ (currentFlightTime - hybridModeStartTime) + " seconds in hybrid mode");
			}
			break;

		case NEURAL_NETWORK:
			// Check if we've completed the full training process
			if (!trainingCompleted && (currentFlightTime - neuralNetworkModeStartTime) >= NN_FLIGHT_SECONDS_THRESHOLD) {
				trainingCompleted = true;

				GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
						"TRAINING PROGRESSION: Training completed! Neural network successfully trained " + "after "
								+ currentFlightTime + " total seconds");
			}
			break;
		}
	}

	/**
	 * Records an action for training purposes.
	 */
	private void recordAction(int action, double confidenceScore) {
		totalActions++;

		// Count action types
		switch (action) {
		case -1:
			upActions++;
			break;
		case 0:
			stayActions++;
			break;
		case 1:
			downActions++;
			break;
		}

		// Track confidence
		totalConfidence += confidenceScore;
		minConfidence = Math.min(minConfidence, confidenceScore);
		confidenceHistory.add(confidenceScore);

		// Record sample for training (convert action to label format)
		if (featureExtractor != null) {
			featureExtractor.recordAction(action + 1); // Convert to 0,1,2 format
		}
	}

	/**
	 * Starts recording flight metrics.
	 */
	public void startRecording() {
		flightStartTime = System.currentTimeMillis();
		currentFlightTime = 0;
		isRecording = true;

		// Reset current flight metrics
		totalActions = 0;
		upActions = 0;
		stayActions = 0;
		downActions = 0;
		totalConfidence = 0.0;
		minConfidence = 1.0;
		confidenceHistory.clear();

		GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL, "Started recording flight metrics");

		// If training is completed, ensure we're in neural network mode
		if (trainingCompleted) {
			setMode(Mode.NEURAL_NETWORK);
			GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
					"Training completed - using NEURAL NETWORK mode");
		}
	}

	/**
	 * Stops recording and processes flight data.
	 */
	public void stopRecording(boolean crashed) {
		if (!isRecording)
			return;

		isRecording = false;
		long endTime = System.currentTimeMillis();
		long flightDuration = (endTime - flightStartTime) / 1000; // In seconds

		// Process flight if it was long enough
		if (flightDuration >= 10) { // Only process flights of 10+ seconds (lowered from 30)
			// Create flight record
			FlightData record = new FlightData(new Date(), flightDuration, totalActions, upActions, stayActions,
					downActions, totalActions > 0 ? totalConfidence / totalActions : 0.0, minConfidence, crashed);

			// Add to history (limit to max size)
			flightHistory.add(record);
			if (flightHistory.size() > MAX_HISTORY) {
				flightHistory.remove(0);
			}

			GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
					"Flight recorded: " + flightDuration + "s, " + (crashed ? "Crashed" : "Successful"));

			// If flight was successful, consider retraining
			if (!crashed) {
				// Commit samples
				featureExtractor.commitSamples();

				// Schedule retraining
				retrainModel();
			} else {
				// Discard the most recent samples before crash
				featureExtractor.discardRecentSamples();

				// If we crashed during training progression, we might need to adjust our
				// progression
				if (!trainingCompleted) {
					handleTrainingCrash(flightDuration);
				}
			}
		} else {
			GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
					"Flight too short (" + flightDuration + "s), not recording");
		}
	}

	/**
	 * Handles a crash during training, potentially adjusting the training
	 * progression.
	 * 
	 * @param flightDuration The duration of the crashed flight
	 */
	private void handleTrainingCrash(long flightDuration) {
	    switch (currentTrainingStage) {
	    case NEURAL_NETWORK:
	        // If we crash early in neural network only mode, or the flight is too short
	        if (flightDuration < NN_FLIGHT_SECONDS_THRESHOLD / 2) {
	            currentTrainingStage = TrainingStage.HYBRID_LEARNING;
	            setMode(Mode.HYBRID);
	            neuralNetworkModeAchieved = false;

	            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
	                    "TRAINING REGRESSION: Reverting to HYBRID mode after short or crashed flight in NN mode");
	        }
	        break;

	    case HYBRID_LEARNING:
	        // If we crash very early in hybrid mode, go back to teacher
	    	System.out.println("Hybrid");
	        if (flightDuration < 20 && flightDuration > NN_FLIGHT_SECONDS_THRESHOLD) {
	            currentTrainingStage = TrainingStage.TEACHER_ONLY;
	            setMode(Mode.TEACHER);
	            hybridModeAchieved = false;
	            confidenceMatchCount = (int) (CONFIDENCE_MATCH_COUNT_THRESHOLD * 0.7); // Don't reset completely

	            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
	                    "TRAINING REGRESSION: Reverting to TEACHER mode after early crash in hybrid mode");
	        } else if (flightDuration > 150 || confidenceMatchCount < CONFIDENCE_MATCH_COUNT_THRESHOLD) {
	        	currentTrainingStage = TrainingStage.NEURAL_NETWORK;
	            setMode(Mode.NEURAL_NETWORK);
	            hybridModeAchieved = false;
	        }
	        break;

	    default:
	        // No regression needed for teacher mode
	        break;
	    }
	}

	/**
	 * Updates the current flight time.
	 */
	public void updateFlightTime(long seconds) {
		if (isRecording) {
			currentFlightTime = seconds;
		}
	}

	/**
	 * Initiates model retraining in a background thread.
	 */
	public boolean retrainModel() {
		if (isTraining) {
			GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
					"Training already in progress");
			return false;
		}

		isTraining = true;
		trainingExecutor.submit(() -> {
			try {
				// Create training set
				MLDataSet trainingSet = featureExtractor.createTrainingSet();

				// Train the network
				boolean success = networkSystem.train(trainingSet);

				if (success) {
					GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
							"Retraining completed successfully");
					
					networkSystem.saveModel();

					// If we were in teacher mode and haven't started the progression yet,
					// reset the confidence match counter to start evaluating the new model
					if (currentTrainingStage == TrainingStage.TEACHER_ONLY && !hybridModeAchieved) {
						confidenceMatchCount = 0;
						GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
								"Starting training progression evaluation with new model");
					}
				} else {
					GameLogger.log(GameLogger.LogLevel.WARNING, GameLogger.LogCategory.NEURAL_NETWORK,
							"Retraining failed - check logs for details");
				}
			} catch (Exception e) {
				GameLogger.logException(GameLogger.LogCategory.NEURAL_NETWORK, "Error during retraining", e);
			} finally {
				isTraining = false;
			}
		});

		return true;
	}

	/**
	 * Manually sets the autopilot mode, overriding training progression.
	 */
	private void setMode(Mode mode) {
		currentMode = mode;
		GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
				"Autopilot mode set to: " + mode);
	}

	/**
	 * Cycles the autopilot mode.
	 */
	public Mode cycleMode() {
		switch (currentMode) {
		case NEURAL_NETWORK:
			setMode(Mode.HYBRID);
			break;
		case HYBRID:
			setMode(Mode.TEACHER);
			break;
		case TEACHER:
			setMode(Mode.NEURAL_NETWORK);
			break;
		}

		return currentMode;
	}

	/**
	 * Suggests an action based on rule-based teacher logic.
	 */
	private int suggestTeacherAction(LinkedList<byte[]> model, int playerRow) {
		// Find tunnel info
		TunnelInfo tunnelInfo = analyzeTunnel(model, playerRow);

		// If we're too close to the top wall, move down
		if (playerRow - tunnelInfo.topWall < 2) {
			return 1; // Move down
		}

		// If we're too close to the bottom wall, move up
		if (tunnelInfo.bottomWall - playerRow < 2) {
			return -1; // Move up
		}

		// If we're significantly above the tunnel center, move down
		if (playerRow < tunnelInfo.center - 1) {
			return 1; // Move down
		}

		// If we're significantly below the tunnel center, move up
		if (playerRow > tunnelInfo.center + 1) {
			return -1; // Move up
		}

		// If the tunnel is trending upward significantly, move up to anticipate
		if (tunnelInfo.trend < -1.5) {
			return -1; // Move up
		}

		// If the tunnel is trending downward significantly, move down to anticipate
		if (tunnelInfo.trend > 1.5) {
			return 1; // Move down
		}

		// If we're in a safe position (near the center), stay
		return 0; // Stay
	}

	/**
	 * Determines if an action would be dangerous.
	 */
	private boolean isDangerousAction(LinkedList<byte[]> model, int playerRow, int action) {
		// Analyze tunnel
		TunnelInfo tunnelInfo = analyzeTunnel(model, playerRow);

		// Calculate new position after action
		int newRow = playerRow + action;

		// Check if action would lead to immediate collision or get dangerously close
		if (newRow <= tunnelInfo.topWall + 1 || newRow >= tunnelInfo.bottomWall - 1) {
			return true;
		}

		// Check if action takes us away from tunnel center when tunnel is narrow
		int distanceToCenter = Math.abs(newRow - tunnelInfo.center);
		int tunnelWidth = tunnelInfo.bottomWall - tunnelInfo.topWall;

		if (tunnelWidth < 5 && distanceToCenter > tunnelWidth / 3) {
			return true;
		}

		// Check if action ignores a significant tunnel trend
		if (tunnelInfo.trend < -2 && action >= 0) { // Tunnel moving up but not moving up
			return true;
		}

		if (tunnelInfo.trend > 2 && action <= 0) { // Tunnel moving down but not moving down
			return true;
		}

		return false;
	}

	/**
	 * Analyzes the tunnel geometry and trajectory.
	 */
	private TunnelInfo analyzeTunnel(LinkedList<byte[]> model, int playerRow) {
		TunnelInfo info = new TunnelInfo();

		// Find immediate tunnel boundaries
		byte[] playerColumn = model.get(PLAYER_COLUMN);

		// Find top edge (first open cell from top)
		info.topWall = 0;
		while (info.topWall < MODEL_HEIGHT && playerColumn[info.topWall] != 0) {
			info.topWall++;
		}

		// Find bottom edge (first open cell from bottom)
		info.bottomWall = MODEL_HEIGHT - 1;
		while (info.bottomWall >= 0 && playerColumn[info.bottomWall] != 0) {
			info.bottomWall--;
		}

		// Calculate center of the current position
		info.center = (info.topWall + info.bottomWall) / 2;

		// Look ahead to determine tunnel trend
		int prevCenter = info.center;
		info.trend = 0;

		for (int j = 1; j <= 5; j++) { // Look 5 steps ahead
			int colIndex = (PLAYER_COLUMN + j) % model.size();
			byte[] column = model.get(colIndex);

			// Find top edge
			int topEdge = 0;
			while (topEdge < MODEL_HEIGHT && column[topEdge] != 0) {
				topEdge++;
			}

			// Find bottom edge
			int bottomEdge = MODEL_HEIGHT - 1;
			while (bottomEdge >= 0 && column[bottomEdge] != 0) {
				bottomEdge--;
			}

			// Calculate center for this column
			if (topEdge <= bottomEdge) {
				int center = (topEdge + bottomEdge) / 2;

				// Calculate weighted trend (closer columns have more influence)
				double weight = 1.0 / j;
				info.trend += (center - prevCenter) * weight;

				prevCenter = center;
			}
		}

		return info;
	}

	/**
	 * Inner class to store tunnel analysis information.
	 */
	private static class TunnelInfo {
		int topWall;
		int bottomWall;
		int center;
		double trend; // Positive = tunnel moving down, negative = tunnel moving up
	}

	// Getters for UI display

	public Mode getCurrentMode() {
		return currentMode;
	}

	public boolean isRecording() {
		return isRecording;
	}

	public boolean isTraining() {
		return isTraining;
	}

	public long getCurrentFlightTime() {
		return currentFlightTime;
	}

	public FlightData getCurrentFlightData() {
		return new FlightData(totalActions, upActions, stayActions, downActions,
				totalActions > 0 ? totalConfidence / totalActions : 0.0, minConfidence,
				new ArrayList<>(confidenceHistory));
	}

	public List<FlightData> getFlightHistory() {
		return new ArrayList<>(flightHistory);
	}

	public void clearTrainingData() {
		if (featureExtractor != null) {
			featureExtractor.clearSamples();
			GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FEATURE_EXTRACTOR, "Training data cleared");
		}

		// Also reset the training progression
		resetTrainingProgression();
		currentMode = Mode.TEACHER;
		currentTrainingStage = TrainingStage.TEACHER_ONLY;

		GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
				"Training progression reset - returning to TEACHER mode");
	}

	/**
	 * Returns whether training has been completed successfully.
	 */
	public boolean isTrainingCompleted() {
		return trainingCompleted;
	}

	/**
	 * Returns the current training stage.
	 */
	public TrainingStage getCurrentTrainingStage() {
		return currentTrainingStage;
	}

	/**
	 * Returns training progression status, useful for UI display.
	 */
	public String getTrainingProgressionStatus() {
		StringBuilder status = new StringBuilder();

		switch (currentTrainingStage) {
		case TEACHER_ONLY:
			status.append("Teacher Learning: ");
			status.append(confidenceMatchCount).append("/").append(CONFIDENCE_MATCH_COUNT_THRESHOLD);
			status.append(" confidence matches");
			break;

		case HYBRID_LEARNING:
			status.append("Hybrid Learning: ");
			long hybridTime = currentFlightTime - hybridModeStartTime;
			status.append(hybridTime).append("/").append(HYBRID_FLIGHT_SECONDS_THRESHOLD);
			status.append(" seconds");
			break;

		case NEURAL_NETWORK:
			if (trainingCompleted) {
				status.append("Training Completed!");
			} else {
				status.append("Neural Network Testing: ");
				long nnTime = currentFlightTime - neuralNetworkModeStartTime;
				status.append(nnTime).append("/").append(NN_FLIGHT_SECONDS_THRESHOLD);
				status.append(" seconds");
			}
			break;
		}

		return status.toString();
	}
}