package ie.atu.sw.network;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;

/**
 * Simplified feature extractor that focuses on core feature extraction
 * and sample recording.
 */
public class SimplifiedFeatureExtractor {
    // Constants
    private static final int FEATURE_COUNT = 10;
    private static final int MODEL_HEIGHT = 20;
    private static final int PLAYER_COLUMN = 15;
    private static final int OUTPUT_COUNT = 3; // Up, stay, down
    private static final int DISCARD_BEFORE_CRASH = 10;
    
    // Last action taken (for labeling)
    private int lastAction = 1; // Default is "stay"
    
    // In-memory storage for training samples
    private final List<double[]> featuresList = new ArrayList<>();
    private final List<double[]> outputsList = new ArrayList<>();
    
    // Buffer for recent samples (to discard on crash)
    private final List<double[]> tempFeaturesList = new ArrayList<>();
    private final List<double[]> tempOutputsList = new ArrayList<>();
    
    // Current feature vector
    private double[] currentFeatures;
    
    /**
     * Extracts features from the current game state.
     * 
     * @param model The linked list representing the game grid
     * @param playerRow The current row position of the player
     * @return An array of normalized feature values
     */
    public double[] extractFeatures(LinkedList<byte[]> model, int playerRow) {
        double[] features = new double[FEATURE_COUNT];
        
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
        double tunnelTrend = 0;
        double tunnelWidth = 0;
        int prevCenter = -1;
        int prevTop = -1;
        int prevBottom = -1;
        int count = 0;
        
        // Track openness above and below separately
        double avgOpenAbove = 0;
        double avgOpenBelow = 0;
        
        // Look ahead to see the tunnel
        int lookAhead = 5;
        
        for (int j = 1; j <= lookAhead; j++) {
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
                
                // Give more weight to closer columns
                double weight = 1.0 / j;
                
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
        
        // Store current features for later use
        currentFeatures = features.clone();
        
        return features;
    }
    
    /**
     * Records the last action taken.
     * 
     * @param action The action taken (0=up, 1=stay, 2=down)
     */
    public void recordAction(int action) {
        // Only record if we have features
        if (currentFeatures == null) return;
        
        // Store features and output in temporary buffer
        tempFeaturesList.add(currentFeatures.clone());
        
        // Create one-hot encoded output
        double[] output = new double[OUTPUT_COUNT];
        output[action] = 1.0;
        
        // Store output in temporary buffer
        tempOutputsList.add(output);
        
        // Update last action
        lastAction = action;
        
        // If the buffer reaches a certain size, commit the oldest samples
        if (tempFeaturesList.size() > DISCARD_BEFORE_CRASH * 2) {
            commitOldestSample();
        }
    }
    
    /**
     * Commits the oldest sample from the temporary buffer to permanent storage.
     */
    private void commitOldestSample() {
        if (!tempFeaturesList.isEmpty()) {
            // Move the oldest sample to permanent storage
            featuresList.add(tempFeaturesList.get(0));
            outputsList.add(tempOutputsList.get(0));
            
            // Remove it from the temporary buffer
            tempFeaturesList.remove(0);
            tempOutputsList.remove(0);
        }
    }
    
    /**
     * Commits all buffered samples to permanent storage.
     */
    public void commitSamples() {
        featuresList.addAll(tempFeaturesList);
        outputsList.addAll(tempOutputsList);
        
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FEATURE_EXTRACTOR,
                      "Committed " + tempFeaturesList.size() + " buffered samples");
        
        // Clear the buffers
        tempFeaturesList.clear();
        tempOutputsList.clear();
    }
    
    /**
     * Discards most recent samples (used when crash occurs).
     */
    public void discardRecentSamples() {
        int discardCount = Math.min(DISCARD_BEFORE_CRASH, tempFeaturesList.size());
        
        if (discardCount > 0) {
            // Keep only samples that happened before potential danger
            List<double[]> keptFeatures = new ArrayList<>(tempFeaturesList.subList(0, tempFeaturesList.size() - discardCount));
            List<double[]> keptOutputs = new ArrayList<>(tempOutputsList.subList(0, tempOutputsList.size() - discardCount));
            
            // Commit kept samples and discard the rest
            featuresList.addAll(keptFeatures);
            outputsList.addAll(keptOutputs);
            
            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FEATURE_EXTRACTOR,
                          "Discarded " + discardCount + " samples, kept " + keptFeatures.size());
            
            // Clear temp buffers
            tempFeaturesList.clear();
            tempOutputsList.clear();
        }
    }
    
    /**
     * Creates a training dataset from collected samples.
     */
    public MLDataSet createTrainingSet() {
        int sampleCount = featuresList.size();
        
        if (sampleCount == 0) {
            GameLogger.log(GameLogger.LogLevel.WARNING, GameLogger.LogCategory.FEATURE_EXTRACTOR,
                           "No training data available");
            return null;
        }
        
        // Create input and ideal arrays for the BasicMLDataSet
        double[][] inputArray = new double[sampleCount][FEATURE_COUNT];
        double[][] idealArray = new double[sampleCount][OUTPUT_COUNT];
        
        // Copy data from lists to arrays
        for (int i = 0; i < sampleCount; i++) {
            inputArray[i] = featuresList.get(i);
            idealArray[i] = outputsList.get(i);
        }
        
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FEATURE_EXTRACTOR,
                       "Creating training set with " + sampleCount + " samples");
        
        // Create and return the dataset
        return new BasicMLDataSet(inputArray, idealArray);
    }
    
    /**
     * Clears all collected training data.
     */
    public void clearSamples() {
        featuresList.clear();
        outputsList.clear();
        tempFeaturesList.clear();
        tempOutputsList.clear();
    }
    
    /**
     * Gets the feature count.
     */
    public int getFeatureCount() {
        return FEATURE_COUNT;
    }
    
    /**
     * Gets the sample count.
     */
    public int getSampleCount() {
        return featuresList.size();
    }
}