package ie.atu.sw.network;

import java.io.File;
import java.util.LinkedList;

import org.encog.engine.network.activation.ActivationSoftMax;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;

/**
 * An optimized neural network controller for plane autopilot.
 * Uses simplified features and a more effective network architecture.
 */
public class NeuralNetworkController {
    // Constants
	private static final String MODEL_FILE = "./resources/optimized_model.eg";
	private int INPUT_NEURONS; // Will be loaded from config
	private int HIDDEN_NEURONS; // Will be loaded from config
	private int MAX_EPOCHS; // Will be loaded from config
	private double TARGET_ERROR; // Will be loaded from config
	private double L2_REGULARIZATION; // Will be loaded from config
	private static final int OUTPUT_NEURONS = 3;
    
    // The neural network
    private BasicNetwork network;
    
    // Feature extractor for getting input data
    private FeatureExtractor featureExtractor;
    
    // Last raw output scores from the network (for logging)
    private double[] lastOutputScores = new double[OUTPUT_NEURONS];
    
    /**
     * Creates a new OptimizedNeuralNetworkController with a reference to the feature extractor.
     * 
     * @param featureExtractor The simplified feature extractor to use
     */
    public NeuralNetworkController(FeatureExtractor featureExtractor) {
        
        // Load configuration
        GameConfig config = GameConfig.getInstance();
        INPUT_NEURONS = config.getInputNeurons();
        HIDDEN_NEURONS = config.getHiddenNeurons();
        MAX_EPOCHS = config.getMaxEpochs();
        TARGET_ERROR = config.getTargetError();
        L2_REGULARIZATION = config.getL2Regularization();
        
        // Ensure INPUT_NEURONS matches the feature count from the extractor
        if (INPUT_NEURONS != featureExtractor.getFeatureCount()) {
            GameLogger.log(GameLogger.LogLevel.WARNING, GameLogger.LogCategory.NEURAL_NETWORK,
                "Mismatch between INPUT_NEURONS (" + INPUT_NEURONS + ") and FEATURE_COUNT (" + 
                featureExtractor.getFeatureCount() + ")");
        }
    	
        this.featureExtractor = featureExtractor;
        
        // Ensure INPUT_NEURONS matches the feature count from the extractor
        if (INPUT_NEURONS != featureExtractor.getFeatureCount()) {
            GameLogger.log(GameLogger.LogLevel.WARNING, GameLogger.LogCategory.NEURAL_NETWORK,
                "Mismatch between INPUT_NEURONS (" + INPUT_NEURONS + ") and FEATURE_COUNT (" + 
                featureExtractor.getFeatureCount() + ")");
        }
        
        // Try to load existing model or create a new one if it doesn't exist
        if (!loadModel()) {
            createNetwork();
        }
        
        // Initialize logging system if not already initialized
        try {
            GameLogger.initialize();
            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK, 
                           "Optimized Neural Network Controller initialized");
        } catch (Exception e) {
            System.err.println("Failed to initialize logging: " + e.getMessage());
        }
    }
    
    /**
     * Creates a new neural network with an optimized topology.
     */
    private void createNetwork() {
        network = new BasicNetwork();
        
        // Input layer
        network.addLayer(new BasicLayer(null, true, INPUT_NEURONS));
        
        // Hidden layers - using TANH activation as in the alternative implementation
        network.addLayer(new BasicLayer(new ActivationTANH(), true, HIDDEN_NEURONS * 2));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, HIDDEN_NEURONS));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, HIDDEN_NEURONS / 2)); // Additional layer
        
        // Output layer (softmax for classification)
        network.addLayer(new BasicLayer(new ActivationSoftMax(), false, OUTPUT_NEURONS));
        
        // Finalize the network structure
        network.getStructure().finalizeStructure();
        network.reset();
        
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
                      "Created optimized neural network with topology: "
                      + INPUT_NEURONS + "-" + (HIDDEN_NEURONS * 2) + "-"+ HIDDEN_NEURONS + "-" + (HIDDEN_NEURONS / 2) + "-" + OUTPUT_NEURONS);
    }
    
    /**
     * Trains the neural network using the collected training data.
     * Uses the in-memory data from SimplifiedFeatureExtractor.
     * 
     * @return true if training was successful, false otherwise
     */
    public boolean trainNetwork() {
        try {
            // Get training data from feature extractor
            MLDataSet trainingSet = featureExtractor.createTrainingSet();
            
            if (trainingSet == null || trainingSet.getRecordCount() < 20) {
                GameLogger.log(GameLogger.LogLevel.WARNING, GameLogger.LogCategory.NEURAL_NETWORK,
                              "Insufficient training data. Need at least 20 samples.");
                return false;
            }
            
            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
                          "Training optimized network with " + trainingSet.getRecordCount() + " samples...");
            
            // Train the network using resilient propagation
            final ResilientPropagation train = new ResilientPropagation(network, trainingSet);
            train.setL2(L2_REGULARIZATION);
            
            long startTime = System.currentTimeMillis();
            int epoch = 0;
            double lastLoggedError = Double.MAX_VALUE;
            double bestError = Double.MAX_VALUE;
            BasicNetwork bestNetwork = null;
            
            // Loop until we hit max epochs or reach target error
            do {
                train.iteration();
                
                // Log progress every 100 epochs or if error improved significantly
                if (epoch % 100 == 0 || train.getError() < lastLoggedError * 0.9) {
                    GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
                                  "Epoch #" + epoch + " Error: " + train.getError());
                    lastLoggedError = train.getError();
                }
                
                // Save best network so far
                if (train.getError() < bestError) {
                    bestError = train.getError();
                    bestNetwork = (BasicNetwork)network.clone();
                }
                
                epoch++;
                
                // Check if we're taking too long
                if ((System.currentTimeMillis() - startTime) > 50000) {
                    GameLogger.log(GameLogger.LogLevel.WARNING, GameLogger.LogCategory.NEURAL_NETWORK,
                                  "Training time limit reached. Stopping at epoch " + epoch);
                    break;
                }
            } while (epoch < MAX_EPOCHS && train.getError() > TARGET_ERROR);
            
            // Stop training
            train.finishTraining();
            
            // Use the best network found during training
            if (bestNetwork != null && bestError < train.getError()) {
                network = bestNetwork;
                GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
                              "Using best network with error: " + bestError);
            }
            
            long trainingTime = System.currentTimeMillis() - startTime;
            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
                          "Training completed in " + (trainingTime / 1000.0) + " seconds. " +
                          "Final error: " + train.getError());
            
            // Evaluate model performance on training set
            evaluateModel(trainingSet);
            
            // Save the trained model
            saveModel();
            
            return true;
            
        } catch (Exception e) {
            GameLogger.logException(GameLogger.LogCategory.NEURAL_NETWORK, 
                                   "Error training network", e);
            return false;
        }
    }
    
    /**
     * Evaluates model performance on a dataset and logs results.
     * 
     * @param dataset The dataset to evaluate on
     */
    private void evaluateModel(MLDataSet dataset) {
        int correct = 0;
        int total = (int) dataset.getRecordCount();
        
        // Confusion matrix (actual vs predicted)
        int[][] confusionMatrix = new int[OUTPUT_NEURONS][OUTPUT_NEURONS];
        
        // Evaluate each record
        for (MLDataPair pair : dataset) {
            MLData input = pair.getInput();
            MLData idealOutput = pair.getIdeal();
            MLData actualOutput = network.compute(input);
            
            // Find highest index in ideal output (expected action)
            int expectedIndex = 0;
            double highestIdeal = idealOutput.getData(0);
            for (int i = 1; i < idealOutput.size(); i++) {
                if (idealOutput.getData(i) > highestIdeal) {
                    highestIdeal = idealOutput.getData(i);
                    expectedIndex = i;
                }
            }
            
            // Find highest index in actual output (predicted action)
            int predictedIndex = 0;
            double highestActual = actualOutput.getData(0);
            for (int i = 1; i < actualOutput.size(); i++) {
                if (actualOutput.getData(i) > highestActual) {
                    highestActual = actualOutput.getData(i);
                    predictedIndex = i;
                }
            }
            
            // Update confusion matrix
            confusionMatrix[expectedIndex][predictedIndex]++;
            
            // Check if prediction was correct
            if (expectedIndex == predictedIndex) {
                correct++;
            }
        }
        
        // Calculate accuracy
        double accuracy = (double) correct / total * 100.0;
        
        // Log overall accuracy
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
                      String.format("Model evaluation: %d/%d correct (%.2f%% accuracy)", 
                                   correct, total, accuracy));
        
        // Log confusion matrix
        StringBuilder cmLog = new StringBuilder("Confusion Matrix:\n");
        cmLog.append("      | UP    | STAY  | DOWN  \n");
        cmLog.append("------|-------|-------|-------\n");
        
        String[] actionNames = {"UP   ", "STAY ", "DOWN "};
        
        for (int i = 0; i < OUTPUT_NEURONS; i++) {
            cmLog.append(actionNames[i]).append(" | ");
            
            for (int j = 0; j < OUTPUT_NEURONS; j++) {
                cmLog.append(String.format("%5d | ", confusionMatrix[i][j]));
            }
            
            cmLog.append("\n");
        }
        
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK, cmLog.toString());
    }
    
    /**
     * Saves the neural network model to a file.
     * 
     * @return true if successful, false otherwise
     */
    public boolean saveModel() {
        try {
            // Make sure directories exist
            File dir = new File("./resources");
            if (!dir.exists()) {
                dir.mkdirs();
            }
            
            // Save the network to a file
            EncogDirectoryPersistence.saveObject(new File(MODEL_FILE), network);
            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
                          "Optimized model saved to " + MODEL_FILE);
            return true;
        } catch (Exception e) {
            GameLogger.logException(GameLogger.LogCategory.NEURAL_NETWORK, 
                                   "Error saving model", e);
            return false;
        }
    }
    
    /**
     * Loads a neural network model from a file.
     * 
     * @return true if successful, false otherwise
     */
    public boolean loadModel() {
        try {
            File modelFile = new File(MODEL_FILE);
            if (modelFile.exists()) {
                network = (BasicNetwork) EncogDirectoryPersistence.loadObject(modelFile);
                GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
                              "Loaded optimized model from " + MODEL_FILE);
                return true;
            }
            return false;
        } catch (Exception e) {
            GameLogger.logException(GameLogger.LogCategory.NEURAL_NETWORK, 
                                   "Error loading model", e);
            return false;
        }
    }
    
    /**
     * Makes a prediction using the neural network and returns detailed output.
     * 
     * @param features The input features
     * @return The predicted action (0=up, 1=stay, 2=down)
     */
    public int predict(double[] features) {
        if (network == null) {
            GameLogger.log(GameLogger.LogLevel.ERROR, GameLogger.LogCategory.NEURAL_NETWORK,
                          "Network not initialized!");
            return 1; // Default to "stay"
        }
        
        // Create the input data array
        MLData input = new BasicMLData(features);
        
        // Get the prediction from the network
        MLData output = network.compute(input);
        
        // Store the raw output scores for logging
        for (int i = 0; i < output.size(); i++) {
            lastOutputScores[i] = output.getData(i);
        }
        
        // Find the index of the highest output (argmax)
        int highestIndex = 0;
        double highestValue = output.getData(0);
        
        for (int i = 1; i < output.size(); i++) {
            if (output.getData(i) > highestValue) {
                highestValue = output.getData(i);
                highestIndex = i;
            }
        }
        
        return highestIndex;
    }
    
    /**
     * Predicts the action to take based on the current game state.
     * 
     * @param model The game grid
     * @param playerRow The current player row
     * @return The action to take (-1=up, 0=stay, 1=down)
     */
    public int predictAction(LinkedList<byte[]> model, int playerRow) {
        // Extract features
        double[] features = featureExtractor.extractFeatures(model, playerRow);
        
        // Log features extraction
        GameLogger.logFeatureExtraction(features, model, playerRow);
        
        // Get prediction
        int prediction = predict(features);
        
        // Convert to game action
        int action = featureExtractor.mapPredictionToAction(prediction);
        
        // Log the neural network decision (without override flag yet)
        GameLogger.logNeuralNetworkDecision(features, lastOutputScores, action, action, playerRow, false, "OPTIMIZED NN");
        
        return action;
    }
    
    /**
     * Gets the last output scores from the network.
     * 
     * @return Array of confidence values for each possible action
     */
    public double[] getLastOutputScores() {
        return lastOutputScores;
    }
}