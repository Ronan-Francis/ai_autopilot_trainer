package ie.atu.sw.network;

import java.io.*;
import java.util.Properties;

/**
 * A configuration system for the game that allows for easy adjustment of parameters
 * without modifying the source code.
 */
public class GameConfig {
    // Singleton instance
    private static GameConfig instance;
    
    // Default config file path
    private static final String CONFIG_FILE_PATH = "./resources/game.properties";
    
    // Properties object to store configuration
    private Properties properties;
    
    // Feature Extractor Configuration
    private int featureCount = 7;
    private int lookAhead = 10;
    private int discardBeforeCrash = 10;
    
    // Neural Network Configuration
    private int inputNeurons = 7;
    private int hiddenNeurons = 20;
    private int maxEpochs = 5000;
    private double targetError = 0.008;
    private double l2Regularization = 0.0003;
    
    // Fuzzy Controller Configuration
    private double confidenceThreshold = 0.65;
    private int minSafeRow = 2;
    private int maxSafeRow = 17; // MODEL_HEIGHT - 3
    private double[] tunnelWeights = {0.6, 0.3, 0.1};
    
    /**
     * Private constructor to enforce singleton pattern.
     */
    private GameConfig() {
        properties = new Properties();
        loadConfig();
    }
    
    /**
     * Gets the singleton instance of GameConfig.
     * 
     * @return The GameConfig instance
     */
    public static GameConfig getInstance() {
        if (instance == null) {
            instance = new GameConfig();
        }
        return instance;
    }
    
    /**
     * Loads configuration from the properties file.
     */
    private void loadConfig() {
        try {
            // Ensure resources directory exists
            File resourcesDir = new File("./resources");
            if (!resourcesDir.exists()) {
                resourcesDir.mkdirs();
            }
            
            File configFile = new File(CONFIG_FILE_PATH);
            
            // If config file doesn't exist, create it with default values
            if (!configFile.exists()) {
                createDefaultConfig();
                GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                              "Created default configuration file at " + CONFIG_FILE_PATH);
            }
            
            // Load properties from file
            try (FileInputStream in = new FileInputStream(configFile)) {
                properties.load(in);
                GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                              "Loaded configuration from " + CONFIG_FILE_PATH);
                
                // Update values from properties
                updateValuesFromProperties();
            }
        } catch (Exception e) {
            GameLogger.log(GameLogger.LogLevel.WARNING, GameLogger.LogCategory.GENERAL,
                          "Error loading configuration: " + e.getMessage() + ". Using default values.");
        }
    }
    
    /**
     * Creates a default configuration file.
     */
    private void createDefaultConfig() throws IOException {
        // Setup default properties
        properties.setProperty("feature.count", String.valueOf(featureCount));
        properties.setProperty("feature.lookAhead", String.valueOf(lookAhead));
        properties.setProperty("feature.discardBeforeCrash", String.valueOf(discardBeforeCrash));
        
        properties.setProperty("neuralnetwork.inputNeurons", String.valueOf(inputNeurons));
        properties.setProperty("neuralnetwork.hiddenNeurons", String.valueOf(hiddenNeurons));
        properties.setProperty("neuralnetwork.maxEpochs", String.valueOf(maxEpochs));
        properties.setProperty("neuralnetwork.targetError", String.valueOf(targetError));
        properties.setProperty("neuralnetwork.l2Regularization", String.valueOf(l2Regularization));
        
        properties.setProperty("fuzzy.confidenceThreshold", String.valueOf(confidenceThreshold));
        properties.setProperty("fuzzy.minSafeRow", String.valueOf(minSafeRow));
        properties.setProperty("fuzzy.maxSafeRow", String.valueOf(maxSafeRow));
        properties.setProperty("fuzzy.tunnelWeights", tunnelWeightsToString());
        
        // Save properties to file
        try (FileOutputStream out = new FileOutputStream(CONFIG_FILE_PATH)) {
            properties.store(out, "Game Configuration");
        }
    }
    
    /**
     * Updates the configuration values from the properties.
     */
    private void updateValuesFromProperties() {
        featureCount = getIntProperty("feature.count", featureCount);
        lookAhead = getIntProperty("feature.lookAhead", lookAhead);
        discardBeforeCrash = getIntProperty("feature.discardBeforeCrash", discardBeforeCrash);
        
        inputNeurons = getIntProperty("neuralnetwork.inputNeurons", inputNeurons);
        hiddenNeurons = getIntProperty("neuralnetwork.hiddenNeurons", hiddenNeurons);
        maxEpochs = getIntProperty("neuralnetwork.maxEpochs", maxEpochs);
        targetError = getDoubleProperty("neuralnetwork.targetError", targetError);
        l2Regularization = getDoubleProperty("neuralnetwork.l2Regularization", l2Regularization);
        
        confidenceThreshold = getDoubleProperty("fuzzy.confidenceThreshold", confidenceThreshold);
        minSafeRow = getIntProperty("fuzzy.minSafeRow", minSafeRow);
        maxSafeRow = getIntProperty("fuzzy.maxSafeRow", maxSafeRow);
        tunnelWeights = stringToTunnelWeights(properties.getProperty("fuzzy.tunnelWeights"));
    }
    
    /**
     * Gets an integer property with a default value.
     */
    private int getIntProperty(String key, int defaultValue) {
        try {
            return Integer.parseInt(properties.getProperty(key, String.valueOf(defaultValue)));
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }
    
    /**
     * Gets a double property with a default value.
     */
    private double getDoubleProperty(String key, double defaultValue) {
        try {
            return Double.parseDouble(properties.getProperty(key, String.valueOf(defaultValue)));
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }
    
    /**
     * Converts tunnel weights array to string for storage.
     */
    private String tunnelWeightsToString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < tunnelWeights.length; i++) {
            sb.append(tunnelWeights[i]);
            if (i < tunnelWeights.length - 1) {
                sb.append(",");
            }
        }
        return sb.toString();
    }
    
    /**
     * Converts string to tunnel weights array.
     */
    private double[] stringToTunnelWeights(String str) {
        if (str == null || str.isEmpty()) {
            return tunnelWeights; // Return default if string is invalid
        }
        
        try {
            String[] parts = str.split(",");
            double[] weights = new double[parts.length];
            for (int i = 0; i < parts.length; i++) {
                weights[i] = Double.parseDouble(parts[i].trim());
            }
            return weights;
        } catch (Exception e) {
            return tunnelWeights; // Return default on error
        }
    }
    
    /**
     * Saves the current configuration to the properties file.
     */
    public void saveConfig() {
        try {
            // Update properties from current values
            properties.setProperty("feature.count", String.valueOf(featureCount));
            properties.setProperty("feature.lookAhead", String.valueOf(lookAhead));
            properties.setProperty("feature.discardBeforeCrash", String.valueOf(discardBeforeCrash));
            
            properties.setProperty("neuralnetwork.inputNeurons", String.valueOf(inputNeurons));
            properties.setProperty("neuralnetwork.hiddenNeurons", String.valueOf(hiddenNeurons));
            properties.setProperty("neuralnetwork.maxEpochs", String.valueOf(maxEpochs));
            properties.setProperty("neuralnetwork.targetError", String.valueOf(targetError));
            properties.setProperty("neuralnetwork.l2Regularization", String.valueOf(l2Regularization));
            
            properties.setProperty("fuzzy.confidenceThreshold", String.valueOf(confidenceThreshold));
            properties.setProperty("fuzzy.minSafeRow", String.valueOf(minSafeRow));
            properties.setProperty("fuzzy.maxSafeRow", String.valueOf(maxSafeRow));
            properties.setProperty("fuzzy.tunnelWeights", tunnelWeightsToString());
            
            // Save properties to file
            try (FileOutputStream out = new FileOutputStream(CONFIG_FILE_PATH)) {
                properties.store(out, "Game Configuration");
                GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                              "Saved configuration to " + CONFIG_FILE_PATH);
            }
        } catch (Exception e) {
            GameLogger.log(GameLogger.LogLevel.ERROR, GameLogger.LogCategory.GENERAL,
                          "Error saving configuration: " + e.getMessage());
        }
    }
    
    // Getters and setters
    
    public int getFeatureCount() {
        return featureCount;
    }
    
    public void setFeatureCount(int featureCount) {
        this.featureCount = featureCount;
    }
    
    public int getLookAhead() {
        return lookAhead;
    }
    
 // Update the config when lookAhead changes
    public void setLookAhead(int lookAhead) {
        this.lookAhead = lookAhead;
        
        // If tunnel weights length doesn't match lookAhead, create a new array
        if (tunnelWeights.length != lookAhead) {
            double[] newWeights = new double[lookAhead];
            
            // Copy existing weights and fill the rest with decreasing values
            for (int i = 0; i < lookAhead; i++) {
                if (i < tunnelWeights.length) {
                    newWeights[i] = tunnelWeights[i];
                } else {
                    // Use a decreasing weight for additional positions
                    newWeights[i] = tunnelWeights[tunnelWeights.length - 1] / (i - tunnelWeights.length + 2);
                }
            }
            
            // Normalize weights to sum to 1
            double sum = 0;
            for (double w : newWeights) sum += w;
            for (int i = 0; i < newWeights.length; i++) {
                newWeights[i] /= sum;
            }
            
            tunnelWeights = newWeights;
        }
    }
    
    public int getDiscardBeforeCrash() {
        return discardBeforeCrash;
    }
    
    public void setDiscardBeforeCrash(int discardBeforeCrash) {
        this.discardBeforeCrash = discardBeforeCrash;
    }
    
    public int getInputNeurons() {
        return inputNeurons;
    }
    
    public void setInputNeurons(int inputNeurons) {
        this.inputNeurons = inputNeurons;
    }
    
    public int getHiddenNeurons() {
        return hiddenNeurons;
    }
    
    public void setHiddenNeurons(int hiddenNeurons) {
        this.hiddenNeurons = hiddenNeurons;
    }
    
    public int getMaxEpochs() {
        return maxEpochs;
    }
    
    public void setMaxEpochs(int maxEpochs) {
        this.maxEpochs = maxEpochs;
    }
    
    public double getTargetError() {
        return targetError;
    }
    
    public void setTargetError(double targetError) {
        this.targetError = targetError;
    }
    
    public double getL2Regularization() {
        return l2Regularization;
    }
    
    public void setL2Regularization(double l2Regularization) {
        this.l2Regularization = l2Regularization;
    }
    
    public double getConfidenceThreshold() {
        return confidenceThreshold;
    }
    
    public void setConfidenceThreshold(double confidenceThreshold) {
        this.confidenceThreshold = confidenceThreshold;
    }
    
    public int getMinSafeRow() {
        return minSafeRow;
    }
    
    public void setMinSafeRow(int minSafeRow) {
        this.minSafeRow = minSafeRow;
    }
    
    public int getMaxSafeRow() {
        return maxSafeRow;
    }
    
    public void setMaxSafeRow(int maxSafeRow) {
        this.maxSafeRow = maxSafeRow;
    }
    
    public double[] getTunnelWeights() {
        return tunnelWeights;
    }
    
    public void setTunnelWeights(double[] tunnelWeights) {
        this.tunnelWeights = tunnelWeights;
    }
}