package ie.atu.sw.game;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.JTabbedPane;
import javax.swing.JTextField;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import ie.atu.sw.network.GameConfig;
import ie.atu.sw.network.GameLogger;
import ie.atu.sw.network.GameLogger.LogCategory;
import ie.atu.sw.network.GameLogger.LogLevel;

/**
 * A simple UI for adjusting game configuration parameters.
 */
public class ConfigUI extends JDialog implements ActionListener, ChangeListener {
    private static final long serialVersionUID = 1L;
    
    // Game configuration
    private GameConfig config;
    
    // Feature extractor controls
    private JSlider featureCountSlider;
    private JSlider lookAheadSlider;
    private JSlider discardBeforeCrashSlider;
    private JLabel featureCountLabel;
    private JLabel lookAheadLabel;
    private JLabel discardBeforeCrashLabel;
    
    // Neural network controls
    private JSlider hiddenNeuronsSlider;
    private JSlider maxEpochsSlider;
    private JTextField targetErrorField;
    private JTextField l2RegularizationField;
    private JLabel hiddenNeuronsLabel;
    private JLabel maxEpochsLabel;
    
    // Fuzzy controller controls
    private JSlider confidenceThresholdSlider;
    private JSlider minSafeRowSlider;
    private JSlider maxSafeRowSlider;
    private JTextField tunnelWeightsField;
    private JLabel confidenceThresholdLabel;
    private JLabel minSafeRowLabel;
    private JLabel maxSafeRowLabel;
    
    // Buttons
    private JButton saveButton;
    private JButton cancelButton;
    private JButton resetButton;
    
    /**
     * Creates a new configuration UI dialog.
     * 
     * @param parent The parent frame
     */
    public ConfigUI(JFrame parent) {
        super(parent, "Game Configuration", true);
        
        // Get game configuration
        config = GameConfig.getInstance();
        
        // Initialize UI components
        initComponents();
        
        // Set dialog properties
        pack();
        setLocationRelativeTo(parent);
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
    }
    
    /**
     * Initializes UI components.
     */
    private void initComponents() {
        // Create tabbed pane
        JTabbedPane tabbedPane = new JTabbedPane();
        
        // Create feature extractor panel
        JPanel featurePanel = createFeaturePanel();
        tabbedPane.addTab("Feature Extractor", featurePanel);
        
        // Create neural network panel
        JPanel nnPanel = createNeuralNetworkPanel();
        tabbedPane.addTab("Neural Network", nnPanel);
        
        // Create fuzzy controller panel
        JPanel fuzzyPanel = createFuzzyPanel();
        tabbedPane.addTab("Fuzzy Controller", fuzzyPanel);
        
        // Create button panel
        JPanel buttonPanel = createButtonPanel();
        
        // Add components to dialog
        getContentPane().setLayout(new BorderLayout());
        getContentPane().add(tabbedPane, BorderLayout.CENTER);
        getContentPane().add(buttonPanel, BorderLayout.SOUTH);
        
        // Update UI with current values
        updateUI();
    }
    
    /**
     * Creates the feature extractor panel.
     */
    private JPanel createFeaturePanel() {
        JPanel panel = new JPanel(new GridLayout(3, 2, 10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        
        // Feature count
        panel.add(new JLabel("Feature Count:"));
        JPanel featureCountPanel = new JPanel(new BorderLayout());
        featureCountSlider = new JSlider(5, 10, config.getFeatureCount());
        featureCountSlider.setMajorTickSpacing(1);
        featureCountSlider.setPaintTicks(true);
        featureCountSlider.setPaintLabels(true);
        featureCountSlider.addChangeListener(this);
        featureCountPanel.add(featureCountSlider, BorderLayout.CENTER);
        featureCountLabel = new JLabel(String.valueOf(config.getFeatureCount()));
        featureCountPanel.add(featureCountLabel, BorderLayout.EAST);
        panel.add(featureCountPanel);
        
        // Look ahead
        panel.add(new JLabel("Look Ahead:"));
        JPanel lookAheadPanel = new JPanel(new BorderLayout());
        lookAheadSlider = new JSlider(2, 8, config.getLookAhead());
        lookAheadSlider.setMajorTickSpacing(1);
        lookAheadSlider.setPaintTicks(true);
        lookAheadSlider.setPaintLabels(true);
        lookAheadSlider.addChangeListener(this);
        lookAheadPanel.add(lookAheadSlider, BorderLayout.CENTER);
        lookAheadLabel = new JLabel(String.valueOf(config.getLookAhead()));
        lookAheadPanel.add(lookAheadLabel, BorderLayout.EAST);
        panel.add(lookAheadPanel);
        
        // Discard before crash
        panel.add(new JLabel("Discard Before Crash:"));
        JPanel discardPanel = new JPanel(new BorderLayout());
        discardBeforeCrashSlider = new JSlider(5, 20, config.getDiscardBeforeCrash());
        discardBeforeCrashSlider.setMajorTickSpacing(5);
        discardBeforeCrashSlider.setPaintTicks(true);
        discardBeforeCrashSlider.setPaintLabels(true);
        discardBeforeCrashSlider.addChangeListener(this);
        discardPanel.add(discardBeforeCrashSlider, BorderLayout.CENTER);
        discardBeforeCrashLabel = new JLabel(String.valueOf(config.getDiscardBeforeCrash()));
        discardPanel.add(discardBeforeCrashLabel, BorderLayout.EAST);
        panel.add(discardPanel);
        
        return panel;
    }
    
    /**
     * Creates the neural network panel.
     */
    private JPanel createNeuralNetworkPanel() {
        JPanel panel = new JPanel(new GridLayout(4, 2, 10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        
        // Hidden neurons
        panel.add(new JLabel("Hidden Neurons:"));
        JPanel hiddenNeuronsPanel = new JPanel(new BorderLayout());
        hiddenNeuronsSlider = new JSlider(10, 50, config.getHiddenNeurons());
        hiddenNeuronsSlider.setMajorTickSpacing(10);
        hiddenNeuronsSlider.setPaintTicks(true);
        hiddenNeuronsSlider.setPaintLabels(true);
        hiddenNeuronsSlider.addChangeListener(this);
        hiddenNeuronsPanel.add(hiddenNeuronsSlider, BorderLayout.CENTER);
        hiddenNeuronsLabel = new JLabel(String.valueOf(config.getHiddenNeurons()));
        hiddenNeuronsPanel.add(hiddenNeuronsLabel, BorderLayout.EAST);
        panel.add(hiddenNeuronsPanel);
        
        // Max epochs
        panel.add(new JLabel("Max Epochs:"));
        JPanel maxEpochsPanel = new JPanel(new BorderLayout());
        maxEpochsSlider = new JSlider(1000, 10000, config.getMaxEpochs());
        maxEpochsSlider.setMajorTickSpacing(3000);
        maxEpochsSlider.setPaintTicks(true);
        maxEpochsSlider.setPaintLabels(true);
        maxEpochsSlider.addChangeListener(this);
        maxEpochsPanel.add(maxEpochsSlider, BorderLayout.CENTER);
        maxEpochsLabel = new JLabel(String.valueOf(config.getMaxEpochs()));
        maxEpochsPanel.add(maxEpochsLabel, BorderLayout.EAST);
        panel.add(maxEpochsPanel);
        
        // Target error
        panel.add(new JLabel("Target Error:"));
        targetErrorField = new JTextField(String.valueOf(config.getTargetError()));
        panel.add(targetErrorField);
        
        // L2 regularization
        panel.add(new JLabel("L2 Regularization:"));
        l2RegularizationField = new JTextField(String.valueOf(config.getL2Regularization()));
        panel.add(l2RegularizationField);
        
        return panel;
    }
    
    /**
     * Creates the fuzzy controller panel.
     */
    private JPanel createFuzzyPanel() {
        JPanel panel = new JPanel(new GridLayout(4, 2, 10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        
        // Confidence threshold
        panel.add(new JLabel("Confidence Threshold:"));
        JPanel confidencePanel = new JPanel(new BorderLayout());
        confidenceThresholdSlider = new JSlider(0, 100, (int)(config.getConfidenceThreshold() * 100));
        confidenceThresholdSlider.setMajorTickSpacing(20);
        confidenceThresholdSlider.setPaintTicks(true);
        confidenceThresholdSlider.setPaintLabels(true);
        confidenceThresholdSlider.addChangeListener(this);
        confidencePanel.add(confidenceThresholdSlider, BorderLayout.CENTER);
        confidenceThresholdLabel = new JLabel(String.format("%.2f", config.getConfidenceThreshold()));
        confidencePanel.add(confidenceThresholdLabel, BorderLayout.EAST);
        panel.add(confidencePanel);
        
        // Min safe row
        panel.add(new JLabel("Min Safe Row:"));
        JPanel minSafeRowPanel = new JPanel(new BorderLayout());
        minSafeRowSlider = new JSlider(0, 10, config.getMinSafeRow());
        minSafeRowSlider.setMajorTickSpacing(2);
        minSafeRowSlider.setPaintTicks(true);
        minSafeRowSlider.setPaintLabels(true);
        minSafeRowSlider.addChangeListener(this);
        minSafeRowPanel.add(minSafeRowSlider, BorderLayout.CENTER);
        minSafeRowLabel = new JLabel(String.valueOf(config.getMinSafeRow()));
        minSafeRowPanel.add(minSafeRowLabel, BorderLayout.EAST);
        panel.add(minSafeRowPanel);
        
        // Max safe row
        panel.add(new JLabel("Max Safe Row:"));
        JPanel maxSafeRowPanel = new JPanel(new BorderLayout());
        maxSafeRowSlider = new JSlider(10, 19, config.getMaxSafeRow());
        maxSafeRowSlider.setMajorTickSpacing(2);
        maxSafeRowSlider.setPaintTicks(true);
        maxSafeRowSlider.setPaintLabels(true);
        maxSafeRowSlider.addChangeListener(this);
        maxSafeRowPanel.add(maxSafeRowSlider, BorderLayout.CENTER);
        maxSafeRowLabel = new JLabel(String.valueOf(config.getMaxSafeRow()));
        maxSafeRowPanel.add(maxSafeRowLabel, BorderLayout.EAST);
        panel.add(maxSafeRowPanel);
        
        // Tunnel weights
        panel.add(new JLabel("Tunnel Weights (comma separated):"));
        StringBuilder sb = new StringBuilder();
        double[] tunnelWeights = config.getTunnelWeights();
        for (int i = 0; i < tunnelWeights.length; i++) {
            sb.append(tunnelWeights[i]);
            if (i < tunnelWeights.length - 1) {
                sb.append(",");
            }
        }
        tunnelWeightsField = new JTextField(sb.toString());
        panel.add(tunnelWeightsField);
        
        return panel;
    }
    
    /**
     * Creates the button panel.
     */
    private JPanel createButtonPanel() {
        JPanel panel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        panel.setBorder(BorderFactory.createEmptyBorder(0, 10, 10, 10));
        
        resetButton = new JButton("Reset to Defaults");
        resetButton.addActionListener(this);
        panel.add(resetButton);
        
        saveButton = new JButton("Save");
        saveButton.addActionListener(this);
        panel.add(saveButton);
        
        cancelButton = new JButton("Cancel");
        cancelButton.addActionListener(this);
        panel.add(cancelButton);
        
        return panel;
    }
    
    /**
     * Updates the UI with the current config values.
     */
    private void updateUI() {
        // Feature extractor
        featureCountSlider.setValue(config.getFeatureCount());
        featureCountLabel.setText(String.valueOf(config.getFeatureCount()));
        
        lookAheadSlider.setValue(config.getLookAhead());
        lookAheadLabel.setText(String.valueOf(config.getLookAhead()));
        
        discardBeforeCrashSlider.setValue(config.getDiscardBeforeCrash());
        discardBeforeCrashLabel.setText(String.valueOf(config.getDiscardBeforeCrash()));
        
        // Neural network
        hiddenNeuronsSlider.setValue(config.getHiddenNeurons());
        hiddenNeuronsLabel.setText(String.valueOf(config.getHiddenNeurons()));
        
        maxEpochsSlider.setValue(config.getMaxEpochs());
        maxEpochsLabel.setText(String.valueOf(config.getMaxEpochs()));
        
        targetErrorField.setText(String.valueOf(config.getTargetError()));
        l2RegularizationField.setText(String.valueOf(config.getL2Regularization()));
        
        // Fuzzy controller
        confidenceThresholdSlider.setValue((int)(config.getConfidenceThreshold() * 100));
        confidenceThresholdLabel.setText(String.format("%.2f", config.getConfidenceThreshold()));
        
        minSafeRowSlider.setValue(config.getMinSafeRow());
        minSafeRowLabel.setText(String.valueOf(config.getMinSafeRow()));
        
        maxSafeRowSlider.setValue(config.getMaxSafeRow());
        maxSafeRowLabel.setText(String.valueOf(config.getMaxSafeRow()));
        
        StringBuilder sb = new StringBuilder();
        double[] tunnelWeights = config.getTunnelWeights();
        for (int i = 0; i < tunnelWeights.length; i++) {
            sb.append(tunnelWeights[i]);
            if (i < tunnelWeights.length - 1) {
                sb.append(",");
            }
        }
        tunnelWeightsField.setText(sb.toString());
    }
    
    /**
     * Saves the configuration values from the UI.
     */
    private void saveConfig() {
        try {
            // Feature extractor
            config.setFeatureCount(featureCountSlider.getValue());
            config.setLookAhead(lookAheadSlider.getValue());
            config.setDiscardBeforeCrash(discardBeforeCrashSlider.getValue());
            
            // Neural network
            config.setHiddenNeurons(hiddenNeuronsSlider.getValue());
            config.setMaxEpochs(maxEpochsSlider.getValue());
            config.setTargetError(Double.parseDouble(targetErrorField.getText()));
            config.setL2Regularization(Double.parseDouble(l2RegularizationField.getText()));
            
            // Fuzzy controller
            config.setConfidenceThreshold(confidenceThresholdSlider.getValue() / 100.0);
            config.setMinSafeRow(minSafeRowSlider.getValue());
            config.setMaxSafeRow(maxSafeRowSlider.getValue());
            
            // Parse tunnel weights
            String[] weightStrs = tunnelWeightsField.getText().split(",");
            double[] weights = new double[weightStrs.length];
            for (int i = 0; i < weightStrs.length; i++) {
                weights[i] = Double.parseDouble(weightStrs[i].trim());
            }
            config.setTunnelWeights(weights);
            
            // Save configuration
            config.saveConfig();
            
            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                          "Configuration saved successfully");
            
            dispose();
        } catch (Exception e) {
            GameLogger.log(GameLogger.LogLevel.ERROR, GameLogger.LogCategory.GENERAL,
                          "Error saving configuration: " + e.getMessage());
        }
    }
    
    /**
     * Handles action events from buttons.
     */
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == saveButton) {
            saveConfig();
        } else if (e.getSource() == cancelButton) {
            dispose();
        } else if (e.getSource() == resetButton) {
            // Create a new config instance with default values
            config = GameConfig.getInstance();
            updateUI();
        }
    }
    
    /**
     * Handles change events from sliders.
     */
    @Override
    public void stateChanged(ChangeEvent e) {
        if (e.getSource() == featureCountSlider) {
            featureCountLabel.setText(String.valueOf(featureCountSlider.getValue()));
        } else if (e.getSource() == lookAheadSlider) {
            lookAheadLabel.setText(String.valueOf(lookAheadSlider.getValue()));
        } else if (e.getSource() == discardBeforeCrashSlider) {
            discardBeforeCrashLabel.setText(String.valueOf(discardBeforeCrashSlider.getValue()));
        } else if (e.getSource() == hiddenNeuronsSlider) {
            hiddenNeuronsLabel.setText(String.valueOf(hiddenNeuronsSlider.getValue()));
        } else if (e.getSource() == maxEpochsSlider) {
            maxEpochsLabel.setText(String.valueOf(maxEpochsSlider.getValue()));
        } else if (e.getSource() == confidenceThresholdSlider) {
            confidenceThresholdLabel.setText(String.format("%.2f", confidenceThresholdSlider.getValue() / 100.0));
        } else if (e.getSource() == minSafeRowSlider) {
            minSafeRowLabel.setText(String.valueOf(minSafeRowSlider.getValue()));
            // Ensure min safe row is less than max safe row
            if (minSafeRowSlider.getValue() >= maxSafeRowSlider.getValue()) {
                maxSafeRowSlider.setValue(minSafeRowSlider.getValue() + 1);
            }
        } else if (e.getSource() == maxSafeRowSlider) {
            maxSafeRowLabel.setText(String.valueOf(maxSafeRowSlider.getValue()));
            // Ensure max safe row is greater than min safe row
            if (maxSafeRowSlider.getValue() <= minSafeRowSlider.getValue()) {
                minSafeRowSlider.setValue(maxSafeRowSlider.getValue() - 1);
            }
        }
    }
}