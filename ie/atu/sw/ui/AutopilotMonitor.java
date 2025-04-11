package ie.atu.sw.ui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.List;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;

import ie.atu.sw.game.AutopilotManager;
import ie.atu.sw.network.AutopilotSystem;
import ie.atu.sw.network.FlightData;
import ie.atu.sw.network.GameLogger;

/**
 * An enhanced visual monitor for the autopilot system that displays real-time
 * statistics, flight history, and controls.
 */
public class AutopilotMonitor extends JFrame implements ActionListener {
    private static final long serialVersionUID = 1L;
    
    // Singleton instance
    private static AutopilotMonitor instance;
    
    // UI Components
    private JLabel modeLabel;
    private JLabel statusLabel;
    private JLabel flightTimeLabel;
    private JLabel totalActionsLabel;
    private JLabel upActionsLabel;
    private JLabel stayActionsLabel;
    private JLabel downActionsLabel;
    private JLabel avgConfidenceLabel;
    private JLabel minConfidenceLabel;
    private JLabel trainingProgressionLabel;
    private JProgressBar trainingProgressBar;
    private ConfidenceHistoryPanel confidenceHistoryPanel;
    private DefaultTableModel flightHistoryModel;
    
    // Timer for refreshing the display
    private Timer refreshTimer;
    private static final int REFRESH_INTERVAL = 500; // ms
    
    private boolean isVisible = false;
    
    /**
     * Gets the singleton instance of AutopilotMonitor.
     */
    public static synchronized AutopilotMonitor getInstance() {
        if (instance == null) {
            instance = new AutopilotMonitor();
        }
        return instance;
    }
    
    /**
     * Private constructor to enforce singleton pattern.
     */
    private AutopilotMonitor() {
        super("Autopilot Performance Monitor");
        
        // Initialize UI
        initializeUI();
        
        // Setup refresh timer
        refreshTimer = new Timer(REFRESH_INTERVAL, this);
        refreshTimer.start();
        
        // Handle window closing
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                isVisible = false;
                refreshTimer.stop();
                setVisible(false);
            }
        });
        
        // Set window properties
        setSize(800, 600);
        setLocationRelativeTo(null);
        setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
    }
    
    /**
     * Initializes the UI components.
     */
    private void initializeUI() {
        // Create tabbed pane for organizing content
        JTabbedPane tabbedPane = new JTabbedPane();
        
        // Create panels for each tab
        JPanel currentFlightPanel = createCurrentFlightPanel();
        JPanel flightHistoryPanel = createFlightHistoryPanel();
        JPanel controlPanel = createControlPanel();
        
        // Add panels to tabbed pane
        tabbedPane.addTab("Current Flight", currentFlightPanel);
        tabbedPane.addTab("Flight History", flightHistoryPanel);
        tabbedPane.addTab("Controls", controlPanel);
        
        // Add tabbed pane to frame
        add(tabbedPane);
    }
    
    /**
     * Creates the panel for displaying current flight information.
     */
    private JPanel createCurrentFlightPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        
        // Status section at top
        JPanel statusPanel = new JPanel(new GridLayout(1, 3));
        statusPanel.setBorder(BorderFactory.createTitledBorder("Status"));
        
        modeLabel = new JLabel("Mode: Unknown", SwingConstants.CENTER);
        statusLabel = new JLabel("Status: Not Recording", SwingConstants.CENTER);
        flightTimeLabel = new JLabel("Flight Time: 00:00", SwingConstants.CENTER);
        
        statusPanel.add(modeLabel);
        statusPanel.add(statusLabel);
        statusPanel.add(flightTimeLabel);
        
        // Training progression section
        JPanel progressionPanel = new JPanel(new BorderLayout());
        progressionPanel.setBorder(BorderFactory.createTitledBorder("Training Progression"));
        
        trainingProgressionLabel = new JLabel("Not in training mode", SwingConstants.CENTER);
        trainingProgressionLabel.setFont(new Font("SansSerif", Font.BOLD, 12));
        
        trainingProgressBar = new JProgressBar(0, 100);
        trainingProgressBar.setStringPainted(true);
        
        progressionPanel.add(trainingProgressionLabel, BorderLayout.NORTH);
        progressionPanel.add(trainingProgressBar, BorderLayout.CENTER);
        
        // Actions section
        JPanel actionsPanel = new JPanel(new GridLayout(2, 3));
        actionsPanel.setBorder(BorderFactory.createTitledBorder("Actions"));
        
        totalActionsLabel = new JLabel("Total: 0", SwingConstants.CENTER);
        upActionsLabel = new JLabel("Up: 0 (0.0%)", SwingConstants.CENTER);
        stayActionsLabel = new JLabel("Stay: 0 (0.0%)", SwingConstants.CENTER);
        downActionsLabel = new JLabel("Down: 0 (0.0%)", SwingConstants.CENTER);
        avgConfidenceLabel = new JLabel("Avg. Confidence: 0.0%", SwingConstants.CENTER);
        minConfidenceLabel = new JLabel("Min. Confidence: 0.0%", SwingConstants.CENTER);
        
        actionsPanel.add(totalActionsLabel);
        actionsPanel.add(upActionsLabel);
        actionsPanel.add(stayActionsLabel);
        actionsPanel.add(downActionsLabel);
        actionsPanel.add(avgConfidenceLabel);
        actionsPanel.add(minConfidenceLabel);
        
        // Confidence history visualization
        confidenceHistoryPanel = new ConfidenceHistoryPanel();
        confidenceHistoryPanel.setPreferredSize(new Dimension(800, 300));
        confidenceHistoryPanel.setBorder(BorderFactory.createTitledBorder("Confidence History"));
        
        // Last neural network decision
        JPanel decisionPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        decisionPanel.setBorder(BorderFactory.createTitledBorder("Last Decision"));
        JLabel lastNNDecisionLabel = new JLabel("Neural Network: " + GameLogger.getLastNeuralNetworkDecision());
        JLabel lastActualDecisionLabel = new JLabel("Actual: " + GameLogger.getLastActualDecision());
        decisionPanel.add(lastNNDecisionLabel);
        decisionPanel.add(lastActualDecisionLabel);
        
        // Assemble layout
        JPanel topPanel = new JPanel(new BorderLayout());
        topPanel.add(statusPanel, BorderLayout.NORTH);
        topPanel.add(progressionPanel, BorderLayout.CENTER);
        topPanel.add(actionsPanel, BorderLayout.SOUTH);
        
        JPanel centerPanel = new JPanel(new BorderLayout());
        centerPanel.add(confidenceHistoryPanel, BorderLayout.CENTER);
        centerPanel.add(decisionPanel, BorderLayout.SOUTH);
        
        panel.add(topPanel, BorderLayout.NORTH);
        panel.add(centerPanel, BorderLayout.CENTER);
        
        return panel;
    }
    
    /**
     * Creates the panel for displaying flight history.
     */
    private JPanel createFlightHistoryPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        
        // Table model for flight history
        String[] columns = {
            "Date", "Duration", "Actions", "Up %", "Stay %", 
            "Down %", "Avg. Conf.", "Min. Conf.", "Result"
        };
        flightHistoryModel = new DefaultTableModel(columns, 0);
        
        // Create table with custom rendering
        JTable flightHistoryTable = new JTable(flightHistoryModel) {
            private static final long serialVersionUID = 1L;
            
            @Override
            public boolean isCellEditable(int row, int column) {
                return false; // Make table non-editable
            }
        };
        
        // Set custom renderers for better display
        DefaultTableCellRenderer centerRenderer = new DefaultTableCellRenderer();
        centerRenderer.setHorizontalAlignment(JLabel.CENTER);
        
        for (int i = 0; i < flightHistoryTable.getColumnCount(); i++) {
            flightHistoryTable.getColumnModel().getColumn(i).setCellRenderer(centerRenderer);
        }
        
        // Add table to a scroll pane
        JScrollPane scrollPane = new JScrollPane(flightHistoryTable);
        
        // Create control buttons
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        JButton refreshButton = new JButton("Refresh");
        refreshButton.addActionListener(e -> refreshFlightHistory());
        buttonPanel.add(refreshButton);
        
        // Add components to panel
        panel.add(new JLabel("Flight History", SwingConstants.CENTER), BorderLayout.NORTH);
        panel.add(scrollPane, BorderLayout.CENTER);
        panel.add(buttonPanel, BorderLayout.SOUTH);
        
        return panel;
    }
    
    /**
     * Creates the control panel.
     */
    private JPanel createControlPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        
        // Mode selection panel
        JPanel modePanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 20));
        modePanel.setBorder(BorderFactory.createTitledBorder("Autopilot Mode"));
        
        JButton nnModeButton = new JButton("Neural Network");
        JButton hybridModeButton = new JButton("Hybrid");
        JButton teacherModeButton = new JButton("Teacher");
        
        // Mode button actions
        nnModeButton.addActionListener(e -> setMode(AutopilotSystem.Mode.NEURAL_NETWORK));
        hybridModeButton.addActionListener(e -> setMode(AutopilotSystem.Mode.HYBRID));
        teacherModeButton.addActionListener(e -> setMode(AutopilotSystem.Mode.TEACHER));
        
        modePanel.add(nnModeButton);
        modePanel.add(hybridModeButton);
        modePanel.add(teacherModeButton);
        
        // Training panel
        JPanel trainingPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 20));
        trainingPanel.setBorder(BorderFactory.createTitledBorder("Training"));
        
        JButton retrainButton = new JButton("Retrain Network");
        retrainButton.addActionListener(e -> {
            // Get AutopilotManager only when needed
            AutopilotManager manager = AutopilotManager.getInstance();
            boolean success = manager.retrainModel();
            if (success) {
                statusLabel.setText("Status: Retraining...");
            }
        });
        
        JButton clearDataButton = new JButton("Clear Training Data");
        clearDataButton.addActionListener(e -> {
            // Get AutopilotManager only when needed
            AutopilotManager manager = AutopilotManager.getInstance();
            manager.clearTrainingData();
            statusLabel.setText("Status: Training Data Cleared");
        });
        
        trainingPanel.add(retrainButton);
        trainingPanel.add(clearDataButton);
        
        // Recording panel
        JPanel recordingPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 20));
        recordingPanel.setBorder(BorderFactory.createTitledBorder("Recording"));
        
        JButton startRecordingButton = new JButton("Start Recording");
        startRecordingButton.addActionListener(e -> {
            // Get AutopilotManager only when needed
            AutopilotManager manager = AutopilotManager.getInstance();
            if (!manager.isRecording()) {
                manager.startRecording();
                statusLabel.setText("Status: Recording");
            }
        });
        
        JButton stopRecordingButton = new JButton("Stop Recording");
        stopRecordingButton.addActionListener(e -> {
            // Get AutopilotManager only when needed
            AutopilotManager manager = AutopilotManager.getInstance();
            if (manager.isRecording()) {
                manager.stopRecording(false);
                statusLabel.setText("Status: Not Recording");
            }
        });
        
        recordingPanel.add(startRecordingButton);
        recordingPanel.add(stopRecordingButton);
        
        // Game control panel
        JPanel gamePanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 20));
        gamePanel.setBorder(BorderFactory.createTitledBorder("Game Control"));
        
        JButton teacherViewButton = new JButton("Open Teacher View");
        teacherViewButton.addActionListener(e -> {
            TeacherPanel.getInstance().toggleVisibility();
        });
        
        gamePanel.add(teacherViewButton);
        
        // Assemble layout
        JPanel controlsContainer = new JPanel(new GridLayout(4, 1, 10, 10));
        controlsContainer.add(modePanel);
        controlsContainer.add(trainingPanel);
        controlsContainer.add(recordingPanel);
        controlsContainer.add(gamePanel);
        
        panel.add(controlsContainer, BorderLayout.CENTER);
        
        return panel;
    }
    
    /**
     * Sets the autopilot mode.
     */
    private void setMode(AutopilotSystem.Mode targetMode) {
        // Get AutopilotManager only when needed
        AutopilotManager manager = AutopilotManager.getInstance();
        
        // Keep cycling until we reach the desired mode
        AutopilotSystem.Mode currentMode = manager.getMode();
        while (currentMode != targetMode) {
            currentMode = manager.cycleMode();
        }
        
        // Update UI
        updateModeDisplay();
    }
    
    /**
     * Updates the mode display.
     */
    private void updateModeDisplay() {
        // Get AutopilotManager only when needed
        AutopilotManager manager = AutopilotManager.getInstance();
        AutopilotSystem.Mode mode = manager.getMode();
        
        if (mode != null) {
            modeLabel.setText("Mode: " + mode);
            
            // Change color based on mode
            switch (mode) {
                case NEURAL_NETWORK:
                    modeLabel.setForeground(Color.BLUE);
                    break;
                case HYBRID:
                    modeLabel.setForeground(new Color(0, 150, 0)); // Dark green
                    break;
                case TEACHER:
                    modeLabel.setForeground(Color.RED);
                    break;
                default:
                    modeLabel.setForeground(Color.BLACK);
            }
        } else {
            modeLabel.setText("Mode: Unknown");
            modeLabel.setForeground(Color.BLACK);
        }
    }
    
    /**
     * Updates the training progression display with the current status.
     */
    private void updateTrainingProgressionDisplay() {
        // Get AutopilotManager and the underlying system
        AutopilotManager manager = AutopilotManager.getInstance();
        try {
            AutopilotSystem system = manager.getAutopilotSystem();
            if (system != null) {
                // Get training status
                boolean isCompleted = system.isTrainingCompleted();
                String progressStatus = system.getTrainingProgressionStatus();
                
                if (isCompleted) {
                    // Training is completed
                    trainingProgressionLabel.setText("Training Completed Successfully!");
                    trainingProgressionLabel.setForeground(new Color(0, 150, 0)); // Dark green
                    trainingProgressBar.setValue(100);
                    trainingProgressBar.setForeground(new Color(0, 150, 0)); // Dark green
                } else {
                    // Training in progress - update with current status
                    trainingProgressionLabel.setText(progressStatus);
                    trainingProgressionLabel.setForeground(Color.BLUE);
                    
                    // Parse the progress from the status string and update progress bar
                    updateProgressBar(progressStatus);
                }
            } else {
                trainingProgressionLabel.setText("Autopilot system not available");
                trainingProgressionLabel.setForeground(Color.RED);
                trainingProgressBar.setValue(0);
            }
        } catch (Exception e) {
            // If we can't get the training status (e.g., method doesn't exist yet)
            trainingProgressionLabel.setText("Training status not available");
            trainingProgressionLabel.setForeground(Color.GRAY);
            trainingProgressBar.setValue(0);
        }
    }
    
    /**
     * Updates the progress bar based on the training status string.
     */
    private void updateProgressBar(String status) {
        try {
            // Different formats based on training stage
            if (status.contains("Teacher Learning:")) {
                // Format: "Teacher Learning: X/Y confidence matches"
                String[] parts = status.split(":");
                String[] values = parts[1].trim().split("/");
                int current = Integer.parseInt(values[0].trim());
                int total = Integer.parseInt(values[1].trim().split(" ")[0]);
                
                trainingProgressBar.setValue((current * 100) / total);
                trainingProgressBar.setForeground(new Color(255, 165, 0)); // Orange
            } 
            else if (status.contains("Hybrid Learning:")) {
                // Format: "Hybrid Learning: X/Y seconds"
                String[] parts = status.split(":");
                String[] values = parts[1].trim().split("/");
                int current = Integer.parseInt(values[0].trim());
                int total = Integer.parseInt(values[1].trim().split(" ")[0]);
                
                trainingProgressBar.setValue((current * 100) / total);
                trainingProgressBar.setForeground(new Color(0, 150, 0)); // Dark green
            }
            else if (status.contains("Neural Network Testing:")) {
                // Format: "Neural Network Testing: X/Y seconds"
                String[] parts = status.split(":");
                String[] values = parts[1].trim().split("/");
                int current = Integer.parseInt(values[0].trim());
                int total = Integer.parseInt(values[1].trim().split(" ")[0]);
                
                trainingProgressBar.setValue((current * 100) / total);
                trainingProgressBar.setForeground(Color.BLUE);
            } 
            else {
                // Default if we can't parse
                trainingProgressBar.setValue(50);
                trainingProgressBar.setForeground(Color.GRAY);
            }
        } catch (Exception e) {
            // If parsing fails, set to indeterminate
            trainingProgressBar.setIndeterminate(true);
        }
    }
    
    /**
     * Refreshes the flight history table.
     */
    private void refreshFlightHistory() {
        // Get AutopilotManager only when needed
        AutopilotManager manager = AutopilotManager.getInstance();
        
        // Clear existing rows
        flightHistoryModel.setRowCount(0);
        
        // Get updated flight history
        List<FlightData> history = manager.getFlightHistory();
        
        if (history != null) {
            // Add each flight to the table in reverse order (newest first)
            for (int i = history.size() - 1; i >= 0; i--) {
                FlightData flight = history.get(i);
                
                // Format the data
                String timestamp = flight.getTimestamp() != null ? 
                        flight.getTimestamp().toString() : "In Progress";
                String duration = FlightData.formatTime(flight.getFlightDuration());
                String actions = String.valueOf(flight.getTotalActions());
                String upPercent = String.format("%.1f%%", flight.getUpPercentage() * 100);
                String stayPercent = String.format("%.1f%%", flight.getStayPercentage() * 100);
                String downPercent = String.format("%.1f%%", flight.getDownPercentage() * 100);
                String avgConf = String.format("%.1f%%", flight.getAvgConfidence() * 100);
                String minConf = String.format("%.1f%%", flight.getMinConfidence() * 100);
                String result = flight.isCrashed() ? "Crashed" : "Success";
                
                // Add row to table
                flightHistoryModel.addRow(new Object[] {
                    timestamp, duration, actions, upPercent, stayPercent, 
                    downPercent, avgConf, minConf, result
                });
            }
        }
    }
    
    /**
     * Updates the current flight display with the latest data.
     */
    private void updateCurrentFlightDisplay() {
        // Get AutopilotManager only when needed
        AutopilotManager manager = AutopilotManager.getInstance();
        
        // Update status
        updateModeDisplay();
        
        // Update training progression display
        updateTrainingProgressionDisplay();
        
        boolean isRecording = manager.isRecording();
        boolean isTraining = manager.isRetraining();
        
        if (isTraining) {
            statusLabel.setText("Status: Training in Progress");
            statusLabel.setForeground(Color.ORANGE);
        } else if (isRecording) {
            statusLabel.setText("Status: Recording");
            statusLabel.setForeground(new Color(0, 150, 0)); // Dark green
        } else {
            statusLabel.setText("Status: Not Recording");
            statusLabel.setForeground(Color.RED);
        }
        
        // Update flight time
        long flightTime = manager.getCurrentFlightTime();
        flightTimeLabel.setText("Flight Time: " + FlightData.formatTime(flightTime));
        
        // Update current flight metrics
        FlightData currentFlight = manager.getCurrentFlightData();
        if (currentFlight != null) {
            totalActionsLabel.setText("Total: " + currentFlight.getTotalActions());
            
            upActionsLabel.setText(String.format("Up: %d (%.1f%%)", 
                    currentFlight.getUpActions(), 
                    currentFlight.getUpPercentage() * 100));
            
            stayActionsLabel.setText(String.format("Stay: %d (%.1f%%)", 
                    currentFlight.getStayActions(), 
                    currentFlight.getStayPercentage() * 100));
            
            downActionsLabel.setText(String.format("Down: %d (%.1f%%)", 
                    currentFlight.getDownActions(), 
                    currentFlight.getDownPercentage() * 100));
            
            avgConfidenceLabel.setText(String.format("Avg. Confidence: %.1f%%", 
                    currentFlight.getAvgConfidence() * 100));
            
            minConfidenceLabel.setText(String.format("Min. Confidence: %.1f%%", 
                    currentFlight.getMinConfidence() * 100));
            
            // Update confidence history chart
            confidenceHistoryPanel.updateData(currentFlight.getConfidenceHistory());
        }
    }
    
    /**
     * Toggles the visibility of the window.
     */
    public void toggleVisibility() {
        isVisible = !isVisible;
        setVisible(isVisible);
        
        if (isVisible) {
            refreshTimer.start();
            updateCurrentFlightDisplay();
            refreshFlightHistory();
        } else {
            refreshTimer.stop();
        }
    }
    
    /**
     * ActionListener for the refresh timer.
     */
    @Override
    public void actionPerformed(ActionEvent e) {
        if (isVisible) {
            SwingUtilities.invokeLater(() -> {
                updateCurrentFlightDisplay();
            });
        }
    }
    
    /**
     * Shows the monitor window.
     * Properly overrides Component.show()
     */
    @Override
    public void show() {
        // Don't call setVisible() directly as it will cause infinite recursion
        // Instead, call super.show() to handle visibility properly
        super.show();
        
        isVisible = true;
        refreshTimer.start();
        updateCurrentFlightDisplay();
        refreshFlightHistory();
    }
}