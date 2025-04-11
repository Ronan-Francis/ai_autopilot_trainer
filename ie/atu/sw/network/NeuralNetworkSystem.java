package ie.atu.sw.network;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.encog.engine.network.activation.ActivationSoftMax;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;

/**
 * Combined neural network system that handles creation, training, and prediction.
 * Replaces both NetworkController and NetworkTrainer.
 */
public class NeuralNetworkSystem {
    // Constants
    private static final String MODEL_FILE = "./resources/network_model.eg";
    private static final int OUTPUT_NEURONS = 3;
    private static final int MAX_EPOCHS = 5000;
    private static final double TARGET_ERROR = 0.008;
    private static final double L2_REGULARIZATION = 0.0003;
    
    // Configuration
    private int inputNeurons;
    private int hiddenNeurons;
    
    // The neural network
    private BasicNetwork network;
    
    // Last raw output scores from the network (for logging)
    private double[] lastOutputScores = new double[OUTPUT_NEURONS];
    
    /**
     * Creates a new neural network system.
     * 
     * @param inputNeurons Number of input neurons (features)
     * @param hiddenNeurons Number of hidden neurons
     */
    public NeuralNetworkSystem(int inputNeurons, int hiddenNeurons) {
        this.inputNeurons = inputNeurons;
        this.hiddenNeurons = hiddenNeurons;
        
        // Try to load existing model or create a new one
        if (!loadModel()) {
            createNetwork();
        }
        
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK, 
                       "Neural Network System initialized");
    }
    
    /**
     * Creates a new neural network with an optimized topology.
     */
    private void createNetwork() {
        network = new BasicNetwork();
        
        // Input layer
        network.addLayer(new BasicLayer(null, true, inputNeurons));
        
        // Hidden layers - using TANH activation
        network.addLayer(new BasicLayer(new ActivationTANH(), true, hiddenNeurons * 2));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, hiddenNeurons));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, hiddenNeurons / 2));
        
        // Output layer (softmax for classification)
        network.addLayer(new BasicLayer(new ActivationSoftMax(), false, OUTPUT_NEURONS));
        
        // Finalize the network structure
        network.getStructure().finalizeStructure();
        network.reset();
        
        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
                      "Created neural network with topology: "
                      + inputNeurons + "-" + (hiddenNeurons * 2) + "-" 
                      + hiddenNeurons + "-" + (hiddenNeurons / 2) + "-" + OUTPUT_NEURONS);
    }
    
    /**
     * Trains the neural network using collected training data.
     * 
     * @param trainingSet The training data set
     * @return true if training was successful, false otherwise
     */
    public boolean train(MLDataSet trainingSet) {
        try {
            // Validate training data
            if (trainingSet == null || trainingSet.getRecordCount() < 20) {
                GameLogger.log(GameLogger.LogLevel.WARNING, GameLogger.LogCategory.NEURAL_NETWORK,
                              "Insufficient training data. Need at least 20 samples.");
                return false;
            }
            
            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
                          "Training network with " + trainingSet.getRecordCount() + " samples...");
            
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
            
            // Save the trained model
            saveModel();
            
            return true;
        } catch (Exception e) {
            GameLogger.logException(GameLogger.LogCategory.NEURAL_NETWORK, 
                                   "Error during training", e);
            return false;
        }
    }
    
    /**
     * Makes a prediction using the neural network.
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
                          "Model saved to " + MODEL_FILE);
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
                              "Loaded model from " + MODEL_FILE);
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
     * Gets the last output scores from the network.
     * 
     * @return Array of confidence values for each possible action
     */
    public double[] getLastOutputScores() {
        return lastOutputScores;
    }
}