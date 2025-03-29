package ie.atu.sw.network;

import java.io.*;
import java.util.*;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;


/**
 * A simplified feature extractor that focuses on the most important
 * game state information for flying decisions.
 */
public class FeatureExtractor {
    // Constants
    private static final int FEATURE_COUNT = 5; // Reduced feature set
    private static final int MODEL_HEIGHT = 20;
    private static final int PLAYER_COLUMN = 15;
    private static final int OUTPUT_COUNT = 3; // Up, stay, down
    
    // Last action taken (for labeling)
    private int lastAction = 1; // Default is "stay"
    
    // In-memory storage for training samples (features and expected outputs)
    private final List<double[]> featuresList = new ArrayList<>();
    private final List<double[]> outputsList = new ArrayList<>();
    
    // Buffer
    private static final int DISCARD_BEFORE_CRASH = 10; // Number of samples to discard before crash
    private final List<double[]> tempFeaturesList = new ArrayList<>(); // Temporary buffer for features
    private final List<double[]> tempOutputsList = new ArrayList<>(); // Temporary buffer for outputs

    /**
     * Constructs a new SimplifiedFeatureExtractor
     */
    public FeatureExtractor() {
        // Ensure resources directory exists
        File resourcesDir = new File("./resources");
        if (!resourcesDir.exists()) {
            resourcesDir.mkdirs();
        }
        
        // Log initialization
        try {
            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FEATURE_EXTRACTOR, 
                          "Simplified feature extractor initialized");
        } catch (Exception e) {
            // If GameLogger isn't initialized yet, ignore this
        }
    }
    
    /**
     * Extracts features from the current game state using an improved approach.
     * Adds additional features to better capture the tunnel trajectory and geometry.
     * 
     * @param model The linked list representing the game grid
     * @param playerRow The current row position of the player
     * @return An array of normalized feature values
     */
    public double[] extractFeatures(LinkedList<byte[]> model, int playerRow) {
        // Increase feature count to capture more information about tunnel geometry
        double[] features = new double[10]; // Expanded from 7 to 10
        
        // Get player column
        byte[] playerColumn = model.get(PLAYER_COLUMN);
        
        // Calculate distance to top obstacle from player
        double immediateTop = 0;
        for (int i = playerRow; i >= 0; i--) {
            if (playerColumn[i] != 0) break;
            immediateTop++;
        }
        
        // Calculate distance to bottom obstacle from player
        double immediateBottom = 0;
        for (int i = playerRow; i < MODEL_HEIGHT; i++) {
            if (playerColumn[i] != 0) break;
            immediateBottom++;
        }
        
        // Calculate average distances and tunnel centers in look-ahead columns
        double avgTop = 0;
        double avgBottom = 0;
        double avgTunnelCenter = 0;
        double tunnelTrend = 0; // Track if tunnel is moving up or down
        double tunnelWidth = 0; // Track tunnel width
        int prevCenter = -1;
        int prevTop = -1;
        int prevBottom = -1;
        int count = 0;
        
        // Track openness above and below separately
        double avgOpenAbove = 0;
        double avgOpenBelow = 0;
        
        // Increase look-ahead to see further
        int extendedLookAhead = 5; // Increased to see further ahead
        
        for (int j = 1; j <= extendedLookAhead; j++) {
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
            
            // If there's a valid tunnel
            if (topEdge <= bottomEdge) {
                // Calculate tunnel center for this column
                int center = (topEdge + bottomEdge) / 2;
                avgTunnelCenter += center;
                
                // Calculate tunnel width
                int width = bottomEdge - topEdge + 1;
                tunnelWidth += width;
                
                // Calculate tunnel trend with directional weighting
                // Give more weight to more immediate columns
                double weight = 1.0 / j; // Higher weight for closer columns
                
                if (prevCenter >= 0) {
                    double trendValue = (center - prevCenter) * weight;
                    tunnelTrend += trendValue;
                    
                    // Also calculate the rate of change of the top and bottom edges
                    if (prevTop >= 0 && prevBottom >= 0) {
                        avgOpenAbove += (topEdge - prevTop) * weight;
                        avgOpenBelow += (prevBottom - bottomEdge) * weight;
                    }
                }
                
                prevCenter = center;
                prevTop = topEdge;
                prevBottom = bottomEdge;
                
                double top = topEdge;
                double bottom = MODEL_HEIGHT - 1 - bottomEdge;
                
                avgTop += top;
                avgBottom += bottom;
                count++;
            }
        }
        
        // Calculate averages
        if (count > 0) {
            avgTop /= count;
            avgBottom /= count;
            avgTunnelCenter /= count;
            tunnelWidth /= count;
        }
        
        // Normalize features to [0, 1] range
        features[0] = immediateTop / MODEL_HEIGHT;
        features[1] = immediateBottom / MODEL_HEIGHT;
        features[2] = avgTop / MODEL_HEIGHT;
        features[3] = avgBottom / MODEL_HEIGHT;
        features[4] = (double) playerRow / MODEL_HEIGHT;
        features[5] = avgTunnelCenter / MODEL_HEIGHT; 
        features[6] = (tunnelTrend + MODEL_HEIGHT/4) / (MODEL_HEIGHT/2); // Tunnel movement trend
        features[7] = tunnelWidth / MODEL_HEIGHT; // Normalized tunnel width
        features[8] = (avgOpenAbove + MODEL_HEIGHT/4) / (MODEL_HEIGHT/2); // Top edge movement
        features[9] = (avgOpenBelow + MODEL_HEIGHT/4) / (MODEL_HEIGHT/2); // Bottom edge movement
        
        // Log the enhanced features
        GameLogger.log(GameLogger.LogLevel.DEBUG, GameLogger.LogCategory.FEATURE_EXTRACTOR,
                       String.format("Enhanced features: Center=%.2f, Trend=%.2f, Width=%.2f, OpenAbove=%.2f, OpenBelow=%.2f",
                                    avgTunnelCenter, tunnelTrend, tunnelWidth, avgOpenAbove, avgOpenBelow));
        
        return features;
    }
    
    /**
     * Records a training sample to a temporary buffer.
     * These samples will only be committed if the flight continues successfully.
     * If a danger flag is provided, it's recorded along with the sample.
     * 
     * @param features The extracted feature vector
     * @param action The action taken (0=up, 1=stay, 2=down)
     * @param isDanger Optional indicator if this sample was recorded during danger
     */
    public synchronized void recordSample(double[] features, int action, boolean... isDanger) {
        // Store features and output in temporary buffer
        tempFeaturesList.add(features.clone());
        
        // Create one-hot encoded output
        double[] output = new double[OUTPUT_COUNT];
        output[action] = 1.0;
        
        // Store output in temporary buffer
        tempOutputsList.add(output);
        
        // Update last action
        lastAction = action;
        
        // Log if this was a decision made during danger
        boolean dangerState = isDanger.length > 0 && isDanger[0];
        if (dangerState) {
            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FEATURE_EXTRACTOR,
                           "Recorded training sample during DANGER state, action: " + 
                           (action == 0 ? "UP" : (action == 1 ? "STAY" : "DOWN")));
        }
        
        // If the buffer reaches a certain size, commit the oldest samples
        if (tempFeaturesList.size() > DISCARD_BEFORE_CRASH * 2) {
            commitOldestSample();
        }
    }
    
    /**
     * Commits the oldest sample from the temporary buffer to permanent storage.
     */
    private synchronized void commitOldestSample() {
        if (!tempFeaturesList.isEmpty()) {
            // Move the oldest sample to permanent storage
            featuresList.add(tempFeaturesList.get(0));
            outputsList.add(tempOutputsList.get(0));
            
            // Remove it from the temporary buffer
            tempFeaturesList.remove(0);
            tempOutputsList.remove(0);
            
            // Log milestone every 50 samples
            if (featuresList.size() % 50 == 0) {
                GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FEATURE_EXTRACTOR,
                               "Collected " + featuresList.size() + " training samples");
            }
        }
    }
    
    /**
     * Commits all buffered samples to permanent storage.
     * Call this on successful flying (e.g. after several seconds of flying).
     */
    public synchronized void commitAllSamples() {
        featuresList.addAll(tempFeaturesList);
        outputsList.addAll(tempOutputsList);
        
        // Log the addition
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FEATURE_EXTRACTOR,
                      "Committed " + tempFeaturesList.size() + " buffered samples");
        
        // Clear the buffers
        tempFeaturesList.clear();
        tempOutputsList.clear();
    }
    
    /**
     * Discards all buffered samples.
     * Call this when the game ends (crash).
     */
    public synchronized void discardBufferedSamples() {
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FEATURE_EXTRACTOR,
                      "Discarded " + tempFeaturesList.size() + " samples before crash");
        
        tempFeaturesList.clear();
        tempOutputsList.clear();
    }

    /**
     * Creates an Encog MLDataSet from the collected samples.
     * 
     * @return An MLDataSet ready for training
     */
    public MLDataSet createTrainingSet() {
        int sampleCount = featuresList.size();
        
        if (sampleCount == 0) {
            GameLogger.log(GameLogger.LogLevel.WARNING, GameLogger.LogCategory.FEATURE_EXTRACTOR,
                           "No training data available");
            return null;
        }
        
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FEATURE_EXTRACTOR,
                       "Creating training set with " + sampleCount + " samples");
        
        // Create input and ideal arrays for the BasicMLDataSet
        double[][] inputArray = new double[sampleCount][FEATURE_COUNT];
        double[][] idealArray = new double[sampleCount][OUTPUT_COUNT];
        
        // Copy data from lists to arrays
        for (int i = 0; i < sampleCount; i++) {
            inputArray[i] = featuresList.get(i);
            idealArray[i] = outputsList.get(i);
        }
        
        // Log distribution of actions in the training set
        int[] actionCounts = new int[OUTPUT_COUNT];
        for (double[] output : outputsList) {
            for (int i = 0; i < output.length; i++) {
                if (output[i] > 0.5) {
                    actionCounts[i]++;
                    break;
                }
            }
        }
        
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FEATURE_EXTRACTOR,
                       String.format("Action distribution: UP=%d (%.1f%%), STAY=%d (%.1f%%), DOWN=%d (%.1f%%)", 
                                    actionCounts[0], (actionCounts[0] * 100.0 / sampleCount),
                                    actionCounts[1], (actionCounts[1] * 100.0 / sampleCount),
                                    actionCounts[2], (actionCounts[2] * 100.0 / sampleCount)));
        
        // Create and return the dataset
        return new BasicMLDataSet(inputArray, idealArray);
    }
    
    /**
     * Maps a keyboard action to a neural network label.
     * 
     * @param keyDirection The key direction (-1=up, 0=stay, 1=down)
     * @return The corresponding label (0=up, 1=stay, 2=down)
     */
    public int mapActionToLabel(int keyDirection) {
        return keyDirection + 1; // Map -1,0,1 to 0,1,2
    }
    
    /**
     * Maps a neural network prediction to a game action.
     * 
     * @param prediction The neural network output (0=up, 1=stay, 2=down)
     * @return The corresponding game action (-1=up, 0=stay, 1=down)
     */
    public int mapPredictionToAction(int prediction) {
        return prediction - 1; // Map 0,1,2 to -1,0,1
    }
    
    /**
     * Getter for the last action.
     * 
     * @return The last action taken
     */
    public int getLastAction() {
        return lastAction;
    }
    
    /**
     * Sets the last action.
     * 
     * @param action The action to set
     */
    public void setLastAction(int action) {
        this.lastAction = action;
    }
    
    /**
     * Returns the number of features being extracted.
     * 
     * @return The feature count
     */
    public int getFeatureCount() {
        return FEATURE_COUNT;
    }
    
    /**
     * Returns the number of samples collected.
     * 
     * @return The sample count
     */
    public int getSampleCount() {
        return featuresList.size();
    }
    
    /**
     * Clears all collected training data.
     */
    public synchronized void clearTrainingData() {
        featuresList.clear();
        outputsList.clear();
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FEATURE_EXTRACTOR,
                      "Training data cleared");
    }
}