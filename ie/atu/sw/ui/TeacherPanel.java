package ie.atu.sw.ui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.LinkedList;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.Timer;

import ie.atu.sw.network.GameLogger;

/**
 * A visual panel that displays teacher rules, current game status, and allows
 * monitoring the autopilot behavior.
 */
public class TeacherPanel extends JFrame implements ActionListener {
    private static final long serialVersionUID = 1L;
    
    // Singleton instance
    private static TeacherPanel instance;
    
    // Constants
    private static final int MODEL_HEIGHT = 20;
    private static final int MODEL_WIDTH = 30;
    private static final int PLAYER_COLUMN = 15;
    private static final int CELL_SIZE = 15;
    private static final int REFRESH_INTERVAL = 100; // ms
    
    // UI Components
    private JLabel statusLabel;
    private JPanel gameVisualizationPanel;
    private JLabel teacherSuggestionLabel;
    private JLabel neuralSuggestionLabel;
    private JLabel finalDecisionLabel;
    
    // Game state
    private LinkedList<byte[]> currentModel;
    private int playerRow = 10;
    
    // Refresh timer
    private Timer refreshTimer;
    
    // Visibility flag
    private boolean isVisible = false;
    
    /**
     * Gets the singleton instance of TeacherPanel.
     */
    public static synchronized TeacherPanel getInstance() {
        if (instance == null) {
            instance = new TeacherPanel();
        }
        return instance;
    }
    
    /**
     * Private constructor to enforce singleton pattern.
     */
    private TeacherPanel() {
        super("Teacher Panel");
        
        // Initialize UI
        initializeUI();
        
        // Set up refresh timer
        refreshTimer = new Timer(REFRESH_INTERVAL, this);
        
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
        
        // Initialize model
        currentModel = new LinkedList<>();
        for (int i = 0; i < MODEL_WIDTH; i++) {
            currentModel.add(new byte[MODEL_HEIGHT]);
        }
    }
    
    /**
     * Initializes the UI components.
     */
    private void initializeUI() {
        // Main panel
        JPanel mainPanel = new JPanel(new BorderLayout());
        
        // Status panel at top
        JPanel statusPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        statusPanel.setBorder(BorderFactory.createTitledBorder("Status"));
        
        statusLabel = new JLabel("Teacher Panel Initialized");
        statusPanel.add(statusLabel);
        
        // Game visualization panel
        gameVisualizationPanel = new JPanel() {
            private static final long serialVersionUID = 1L;
            
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                drawGameVisualization(g);
            }
        };
        gameVisualizationPanel.setPreferredSize(new Dimension(MODEL_WIDTH * CELL_SIZE, MODEL_HEIGHT * CELL_SIZE));
        gameVisualizationPanel.setBorder(BorderFactory.createTitledBorder("Game Visualization"));
        gameVisualizationPanel.setBackground(Color.WHITE);
        
        // Decision panel
        JPanel decisionPanel = new JPanel(new GridLayout(3, 1));
        decisionPanel.setBorder(BorderFactory.createTitledBorder("Decision Analysis"));
        
        teacherSuggestionLabel = new JLabel("Teacher Suggestion: UNKNOWN");
        neuralSuggestionLabel = new JLabel("Neural Network: UNKNOWN");
        finalDecisionLabel = new JLabel("Final Decision: UNKNOWN");
        
        decisionPanel.add(teacherSuggestionLabel);
        decisionPanel.add(neuralSuggestionLabel);
        decisionPanel.add(finalDecisionLabel);
        
        // Teacher rules panel
        JPanel rulesPanel = new JPanel(new BorderLayout());
        rulesPanel.setBorder(BorderFactory.createTitledBorder("Teacher Rules"));
        
        String rulesText = "<html>" +
                "<p><b>Rule 1:</b> If too close to top wall (< 2 units), move DOWN</p>" +
                "<p><b>Rule 2:</b> If too close to bottom wall (< 2 units), move UP</p>" +
                "<p><b>Rule 3:</b> If significantly above tunnel center (> 1 unit), move DOWN</p>" +
                "<p><b>Rule 4:</b> If significantly below tunnel center (> 1 unit), move UP</p>" +
                "<p><b>Rule 5:</b> If tunnel trend is upward (< -1.5), move UP to anticipate</p>" +
                "<p><b>Rule 6:</b> If tunnel trend is downward (> 1.5), move DOWN to anticipate</p>" +
                "<p><b>Rule 7:</b> If in safe position near center, STAY</p>" +
                "</html>";
        
        JLabel rulesLabel = new JLabel(rulesText);
        rulesPanel.add(rulesLabel, BorderLayout.CENTER);
        
        // Danger conditions panel
        JPanel dangerPanel = new JPanel(new BorderLayout());
        dangerPanel.setBorder(BorderFactory.createTitledBorder("Danger Conditions"));
        
        String dangerText = "<html>" +
                "<p><b>Danger 1:</b> Action would lead to immediate collision</p>" +
                "<p><b>Danger 2:</b> Action would get dangerously close to wall (≤ 1 unit)</p>" +
                "<p><b>Danger 3:</b> Action takes away from center when tunnel is narrow</p>" +
                "<p><b>Danger 4:</b> Action ignores significant tunnel trend</p>" +
                "</html>";
        
        JLabel dangerLabel = new JLabel(dangerText);
        dangerPanel.add(dangerLabel, BorderLayout.CENTER);
        
        // Control panel
        JPanel controlPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        
        JButton closeButton = new JButton("Close");
        closeButton.addActionListener(e -> toggleVisibility());
        
        controlPanel.add(closeButton);
        
        // Assemble panels
        JPanel centerPanel = new JPanel(new BorderLayout());
        centerPanel.add(gameVisualizationPanel, BorderLayout.CENTER);
        centerPanel.add(decisionPanel, BorderLayout.SOUTH);
        
        JPanel rightPanel = new JPanel(new GridLayout(2, 1));
        rightPanel.add(rulesPanel);
        rightPanel.add(dangerPanel);
        
        mainPanel.add(statusPanel, BorderLayout.NORTH);
        mainPanel.add(centerPanel, BorderLayout.CENTER);
        mainPanel.add(rightPanel, BorderLayout.EAST);
        mainPanel.add(controlPanel, BorderLayout.SOUTH);
        
        // Add main panel to frame
        add(mainPanel);
    }
    
    /**
     * Draws the game visualization.
     */
    private void drawGameVisualization(Graphics g) {
        Graphics2D g2 = (Graphics2D) g;
        
        // Draw grid
        for (int x = 0; x < MODEL_WIDTH; x++) {
            for (int y = 0; y < MODEL_HEIGHT; y++) {
                int x1 = x * CELL_SIZE;
                int y1 = y * CELL_SIZE;
                
                // Draw cell border
                g2.setColor(Color.LIGHT_GRAY);
                g2.drawRect(x1, y1, CELL_SIZE, CELL_SIZE);
                
                // Fill obstacles
                if (currentModel != null && x < currentModel.size() && 
                        currentModel.get(x) != null && y < currentModel.get(x).length) {
                    if (currentModel.get(x)[y] != 0) {
                        g2.setColor(Color.BLACK);
                        g2.fillRect(x1, y1, CELL_SIZE, CELL_SIZE);
                    }
                }
                
                // Highlight player position
                if (x == PLAYER_COLUMN && y == playerRow) {
                    g2.setColor(Color.RED);
                    g2.fillRect(x1, y1, CELL_SIZE, CELL_SIZE);
                }
            }
        }
        
        // Highlight next few columns for trajectory visualization
        for (int j = 1; j <= 5; j++) {
            int colIndex = (PLAYER_COLUMN + j) % MODEL_WIDTH;
            
            // Find tunnel boundaries
            int topWall = 0;
            int bottomWall = MODEL_HEIGHT - 1;
            
            if (currentModel != null && colIndex < currentModel.size()) {
                byte[] column = currentModel.get(colIndex);
                
                // Find top edge
                while (topWall < MODEL_HEIGHT && column[topWall] != 0) {
                    topWall++;
                }
                
                // Find bottom edge
                while (bottomWall >= 0 && column[bottomWall] != 0) {
                    bottomWall--;
                }
                
                // Calculate center
                int center = (topWall + bottomWall) / 2;
                
                // Mark tunnel center
                int x1 = colIndex * CELL_SIZE;
                int y1 = center * CELL_SIZE;
                
                g2.setColor(new Color(0, 0, 255, 100)); // Semi-transparent blue
                g2.fillRect(x1, y1, CELL_SIZE, CELL_SIZE);
            }
        }
        
        // Mark current tunnel center
        byte[] playerColumn = currentModel.get(PLAYER_COLUMN);
        int topWall = 0;
        int bottomWall = MODEL_HEIGHT - 1;
        
        // Find top edge
        while (topWall < MODEL_HEIGHT && playerColumn[topWall] != 0) {
            topWall++;
        }
        
        // Find bottom edge
        while (bottomWall >= 0 && playerColumn[bottomWall] != 0) {
            bottomWall--;
        }
        
        // Calculate center
        int center = (topWall + bottomWall) / 2;
        
        int x1 = PLAYER_COLUMN * CELL_SIZE;
        int y1 = center * CELL_SIZE;
        
        g2.setColor(new Color(0, 255, 0, 150)); // Semi-transparent green
        g2.fillRect(x1, y1, CELL_SIZE, CELL_SIZE);
        
        // Add legend
        int legendX = 10;
        int legendY = MODEL_HEIGHT * CELL_SIZE - 70;
        
        g2.setColor(Color.BLACK);
        g2.setFont(new Font("SansSerif", Font.PLAIN, 10));
        
        g2.setColor(Color.RED);
        g2.fillRect(legendX, legendY, 10, 10);
        g2.setColor(Color.BLACK);
        g2.drawString("Player", legendX + 15, legendY + 9);
        
        legendY += 15;
        g2.setColor(new Color(0, 255, 0, 150));
        g2.fillRect(legendX, legendY, 10, 10);
        g2.setColor(Color.BLACK);
        g2.drawString("Current Tunnel Center", legendX + 15, legendY + 9);
        
        legendY += 15;
        g2.setColor(new Color(0, 0, 255, 100));
        g2.fillRect(legendX, legendY, 10, 10);
        g2.setColor(Color.BLACK);
        g2.drawString("Future Tunnel Centers", legendX + 15, legendY + 9);
        
        legendY += 15;
        g2.setColor(Color.BLACK);
        g2.fillRect(legendX, legendY, 10, 10);
        g2.drawString("Obstacle", legendX + 15, legendY + 9);
    }
    
    /**
     * Updates the visualization with the latest game state.
     * 
     * @param model The current game model
     * @param playerRow The current player row
     */
    public void updateGameState(LinkedList<byte[]> model, int playerRow) {
        if (model != null) {
            // Make a deep copy of the model to avoid concurrent modification
            currentModel = new LinkedList<>();
            for (byte[] column : model) {
                currentModel.add(column.clone());
            }
            
            this.playerRow = playerRow;
            
            if (isVisible) {
                // Update decisions panel with latest log info
                neuralSuggestionLabel.setText("Neural Network: " + GameLogger.getLastNeuralNetworkDecision());
                finalDecisionLabel.setText("Final Decision: " + GameLogger.getLastActualDecision());
                
                // Derive teacher suggestion (simplified version)
                teacherSuggestionLabel.setText("Teacher Suggestion: " + deriveTeacherSuggestion());
                
                // Update visualization
                gameVisualizationPanel.repaint();
            }
        }
    }
    
    /**
     * Derives a simplified teacher suggestion based on current state.
     */
    private String deriveTeacherSuggestion() {
        if (currentModel == null || currentModel.isEmpty()) {
            return "UNKNOWN";
        }
        
        byte[] playerColumn = currentModel.get(PLAYER_COLUMN);
        
        // Find tunnel boundaries
        int topWall = 0;
        int bottomWall = MODEL_HEIGHT - 1;
        
        // Find top edge
        while (topWall < MODEL_HEIGHT && playerColumn[topWall] != 0) {
            topWall++;
        }
        
        // Find bottom edge
        while (bottomWall >= 0 && playerColumn[bottomWall] != 0) {
            bottomWall--;
        }
        
        // Calculate center
        int center = (topWall + bottomWall) / 2;
        
        // Calculate tunnel trend
        double trend = 0;
        int prevCenter = center;
        
        for (int j = 1; j <= 5; j++) {
            int colIndex = (PLAYER_COLUMN + j) % MODEL_WIDTH;
            byte[] column = currentModel.get(colIndex);
            
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
                int futureCenter = (topEdge + bottomEdge) / 2;
                
                // Calculate weighted trend (closer columns have more influence)
                double weight = 1.0 / j;
                trend += (futureCenter - prevCenter) * weight;
                
                prevCenter = futureCenter;
            }
        }
        
        // Apply rules
        if (playerRow - topWall < 2) {
            return "DOWN (too close to top wall)";
        }
        
        if (bottomWall - playerRow < 2) {
            return "UP (too close to bottom wall)";
        }
        
        if (playerRow < center - 1) {
            return "DOWN (above tunnel center)";
        }
        
        if (playerRow > center + 1) {
            return "UP (below tunnel center)";
        }
        
        if (trend < -1.5) {
            return "UP (tunnel trending upward)";
        }
        
        if (trend > 1.5) {
            return "DOWN (tunnel trending downward)";
        }
        
        return "STAY (safe position)";
    }
    
    /**
     * Toggles the visibility of the window.
     */
    public void toggleVisibility() {
        isVisible = !isVisible;
        setVisible(isVisible);
        
        if (isVisible) {
            refreshTimer.start();
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
            // Get latest neural network and actual decisions from GameLogger
            SwingUtilities.invokeLater(() -> {
                neuralSuggestionLabel.setText("Neural Network: " + GameLogger.getLastNeuralNetworkDecision());
                finalDecisionLabel.setText("Final Decision: " + GameLogger.getLastActualDecision());
                statusLabel.setText("Monitoring autopilot decisions...");
            });
        }
    }
    
    /**
     * Shows a visualization of the game state.
     */
    public void showVisualization(LinkedList<byte[]> model, int playerRow) {
        updateGameState(model, playerRow);
        
        if (!isVisible) {
            toggleVisibility();
        }
    }
}