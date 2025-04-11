package ie.atu.sw.network;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.Properties;

/**
 * Simplified configuration system for the game.
 */
public class GameConfig {
    // Singleton instance
    private static GameConfig instance;
    
    // Default config file path
    private static final String CONFIG_FILE_PATH = "./resources/game.properties";
    
    // Properties object
    private Properties properties;
    
    // Core configurations with sensible defaults
    private int featureCount = 10;
    private int hiddenNeurons = 20;
    private int discardBeforeCrash = 10;
    private double confidenceThreshold = 0.65;
    private int minSafeRow = 2;
    private int maxSafeRow = 17;
    
    /**
     * Private constructor to enforce singleton pattern.
     */
    private GameConfig() {
        properties = new Properties();
        loadConfig();
    }
    
    /**
     * Gets the singleton instance of GameConfig.
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
            }
            
            // Load properties from file
            try (FileInputStream in = new FileInputStream(configFile)) {
                properties.load(in);
                
                // Update values from properties
                featureCount = getIntProperty("feature.count", featureCount);
                hiddenNeurons = getIntProperty("neuralnetwork.hiddenNeurons", hiddenNeurons);
                discardBeforeCrash = getIntProperty("feature.discardBeforeCrash", discardBeforeCrash);
                confidenceThreshold = getDoubleProperty("fuzzy.confidenceThreshold", confidenceThreshold);
                minSafeRow = getIntProperty("fuzzy.minSafeRow", minSafeRow);
                maxSafeRow = getIntProperty("fuzzy.maxSafeRow", maxSafeRow);
            }
        } catch (Exception e) {
            System.err.println("Error loading configuration: " + e.getMessage());
        }
    }
    
    /**
     * Creates a default configuration file.
     */
    private void createDefaultConfig() {
        try {
            // Setup default properties
            properties.setProperty("feature.count", String.valueOf(featureCount));
            properties.setProperty("neuralnetwork.hiddenNeurons", String.valueOf(hiddenNeurons));
            properties.setProperty("feature.discardBeforeCrash", String.valueOf(discardBeforeCrash));
            properties.setProperty("fuzzy.confidenceThreshold", String.valueOf(confidenceThreshold));
            properties.setProperty("fuzzy.minSafeRow", String.valueOf(minSafeRow));
            properties.setProperty("fuzzy.maxSafeRow", String.valueOf(maxSafeRow));
            
            // Save properties to file
            try (FileOutputStream out = new FileOutputStream(CONFIG_FILE_PATH)) {
                properties.store(out, "Game Configuration");
            }
        } catch (Exception e) {
            System.err.println("Error creating default configuration: " + e.getMessage());
        }
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
     * Saves the current configuration to the properties file.
     */
    public void saveConfig() {
        try {
            // Update properties from current values
            properties.setProperty("feature.count", String.valueOf(featureCount));
            properties.setProperty("neuralnetwork.hiddenNeurons", String.valueOf(hiddenNeurons));
            properties.setProperty("feature.discardBeforeCrash", String.valueOf(discardBeforeCrash));
            properties.setProperty("fuzzy.confidenceThreshold", String.valueOf(confidenceThreshold));
            properties.setProperty("fuzzy.minSafeRow", String.valueOf(minSafeRow));
            properties.setProperty("fuzzy.maxSafeRow", String.valueOf(maxSafeRow));
            
            // Save properties to file
            try (FileOutputStream out = new FileOutputStream(CONFIG_FILE_PATH)) {
                properties.store(out, "Game Configuration");
            }
        } catch (Exception e) {
            System.err.println("Error saving configuration: " + e.getMessage());
        }
    }
    
    // Getters and setters
    public int getFeatureCount() { return featureCount; }
    public void setFeatureCount(int value) { featureCount = value; }
    
    public int getHiddenNeurons() { return hiddenNeurons; }
    public void setHiddenNeurons(int value) { hiddenNeurons = value; }
    
    public int getDiscardBeforeCrash() { return discardBeforeCrash; }
    public void setDiscardBeforeCrash(int value) { discardBeforeCrash = value; }
    
    public double getConfidenceThreshold() { return confidenceThreshold; }
    public void setConfidenceThreshold(double value) { confidenceThreshold = value; }
    
    public int getMinSafeRow() { return minSafeRow; }
    public void setMinSafeRow(int value) { minSafeRow = value; }
    
    public int getMaxSafeRow() { return maxSafeRow; }
    public void setMaxSafeRow(int value) { maxSafeRow = value; }
}