package ie.atu.sw.game;

import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.util.concurrent.ThreadLocalRandom.current;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.stream.Collectors;

import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import ie.atu.sw.network.AutopilotSystem;
import ie.atu.sw.network.GameLogger;
import ie.atu.sw.ui.UIManager;

public class GameView extends JPanel implements ActionListener {
    private static final long serialVersionUID = 1L;
    private static final int MODEL_WIDTH = 30;
    private static final int MODEL_HEIGHT = 20;
    private static final int SCALING_FACTOR = 30;
    private static final int MIN_TOP = 2;
    private static final int MIN_BOTTOM = 18;
    private static final int PLAYER_COLUMN = 15;
    private static final int TIMER_INTERVAL = 100;
    private static final long AUTO_RESTART_DELAY = 3000; // 3 seconds delay for auto restart

    private static final byte ONE_SET = 1;
    private static final byte ZERO_SET = 0;

    private LinkedList<byte[]> model = new LinkedList<>();
    private int prevTop = MIN_TOP;
    private int prevBot = MIN_BOTTOM;
    private javax.swing.Timer timer;
    private long time;
    private int playerRow = 11;
    private int index = MODEL_WIDTH - 1;
    private Dimension dim;

    private Font font = new Font("Dialog", Font.BOLD, 50);
    private Font over = new Font("Dialog", Font.BOLD, 100);
    private Font debugFont = new Font("Monospaced", Font.PLAIN, 14);

    private Sprite sprite;
    private Sprite dyingSprite;

    // Remove local autopilot flag—always use the manager.
    private AutopilotManager autopilotManager;

    private boolean showVisualizations = false;
    private boolean gameOverProcessed = false;
    private boolean showDebugInfo = true;
    private boolean showConfidenceGraph = true;

    public GameView(boolean autopilotEnabled) throws Exception {
        setBackground(Color.LIGHT_GRAY);
        setDoubleBuffered(true);

        // Set up the dimensions
        dim = new Dimension(MODEL_WIDTH * SCALING_FACTOR, MODEL_HEIGHT * SCALING_FACTOR);
        super.setPreferredSize(dim);
        super.setMinimumSize(dim);
        super.setMaximumSize(dim);

        // Initialize the game model
        initModel();

        // Get the AutopilotManager instance
        autopilotManager = AutopilotManager.getInstance();
        
        // Initialize in a sequence that avoids circular dependencies
        if (autopilotEnabled) {
            // First initialize core components
            autopilotManager.initializeComponents();
            
            // Then complete any UI initialization that might have been deferred
            autopilotManager.completeInitialization();
            
            // Finally start recording
            autopilotManager.startRecording();
        }

        // Start the game timer
        timer = new javax.swing.Timer(TIMER_INTERVAL, this);
        timer.start();
    }

    private void initModel() {
        for (int i = 0; i < MODEL_WIDTH; i++) {
            model.add(new byte[MODEL_HEIGHT]);
        }
    }

    public void setSprite(Sprite s) {
        this.sprite = s;
    }

    public void setDyingSprite(Sprite s) {
        this.dyingSprite = s;
    }

    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;

        g2.setColor(Color.LIGHT_GRAY);
        g2.fillRect(0, 0, dim.width, dim.height);

        int x1 = 0, y1 = 0;
        for (int x = 0; x < MODEL_WIDTH; x++) {
            for (int y = 0; y < MODEL_HEIGHT; y++) {
                x1 = x * SCALING_FACTOR;
                y1 = y * SCALING_FACTOR;

                if (model.get(x)[y] != 0) {
                    if (y == playerRow && x == PLAYER_COLUMN) {
                        timer.stop(); // Crash...
                    }
                    g2.setColor(Color.BLACK);
                    g2.fillRect(x1, y1, SCALING_FACTOR, SCALING_FACTOR);
                }

                if (x == PLAYER_COLUMN && y == playerRow) {
                    if (timer.isRunning()) {
                        g2.drawImage(sprite.getNext(), x1, y1, null);
                    } else {
                        g2.drawImage(dyingSprite.getNext(), x1, y1, null);
                    }
                }
            }
        }

        g2.setFont(font);
        g2.setColor(Color.RED);
        g2.fillRect(1 * SCALING_FACTOR, 15 * SCALING_FACTOR, 400, 3 * SCALING_FACTOR);
        g2.setColor(Color.WHITE);
        g2.drawString("Time: " + (int) (time * (TIMER_INTERVAL / 1000.0d)) + "s", 1 * SCALING_FACTOR + 10,
                (15 * SCALING_FACTOR) + (2 * SCALING_FACTOR));

        if (showDebugInfo) {
            drawDebugInfo(g2);
        }

        if (!timer.isRunning() && !gameOverProcessed) {
            g2.setFont(over);
            g2.setColor(Color.RED);
            g2.drawString("Game Over!", (MODEL_WIDTH / 5) * SCALING_FACTOR, (MODEL_HEIGHT / 2) * SCALING_FACTOR);

            // Stop recording via the manager and schedule a restart.
            if (autopilotManager != null) {
                autopilotManager.stopRecording(true);
                
                // Update UI state with crash
                UIManager.getInstance().updateGameState(model, playerRow);
                
                gameOverProcessed = true;
                GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
                        "Flight ended. Auto-restarting...");
                Timer autoRestartTimer = new Timer();
                autoRestartTimer.schedule(new TimerTask() {
                    @Override
                    public void run() {
                        SwingUtilities.invokeLater(() -> reset());
                    }
                }, AUTO_RESTART_DELAY);
                
                // Only retrain if NOT in neural network mode AND flight time is sufficient
                if (autopilotManager.getMode() != AutopilotSystem.Mode.NEURAL_NETWORK) {
                    // Change from 300 (30 seconds) to a more descriptive constant
                    long MINIMUM_FLIGHT_TIME_FOR_TRAINING = 300; // 30 seconds at 100ms intervals
                    if (time >= MINIMUM_FLIGHT_TIME_FOR_TRAINING) {
                        autopilotManager.retrainModel();
                    } else {
                        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
                                "Flight too short for training: " + time + " < " + MINIMUM_FLIGHT_TIME_FOR_TRAINING);
                    }
                } else {
                    GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
                            "Skipping retraining in NEURAL_NETWORK mode");
                }
            }
        }
    }

    private void drawDebugInfo(Graphics2D g2) {
        g2.setFont(debugFont);
        int controlPanelWidth = 300;
        int controlPanelHeight = 150;
        int controlPanelX = 30;
        int controlPanelY = 30;

        g2.setColor(new Color(0, 0, 0, 180));
        g2.fillRect(controlPanelX, controlPanelY, controlPanelWidth, controlPanelHeight);

        g2.setColor(Color.WHITE);
        String modeText = "MODE: ";
        if (autopilotManager != null) {
            AutopilotSystem.Mode mode = autopilotManager.getMode();
            if (mode != null) {
                switch (mode) {
                    case NEURAL_NETWORK:
                        modeText += "NEURAL NETWORK";
                        g2.setColor(Color.CYAN);
                        break;
                    case HYBRID:
                        modeText += "NEURAL NETWORK + TEACHER";
                        g2.setColor(Color.GREEN);
                        break;
                    case TEACHER:
                        modeText += "TEACHER ONLY";
                        g2.setColor(Color.YELLOW);
                        break;
                }
            }
        } else {
            modeText += "UNKNOWN";
        }
        g2.drawString(modeText, controlPanelX + 10, controlPanelY + 20);
        g2.setColor(Color.WHITE);

        g2.drawString("NEURAL NETWORK: " + GameLogger.getLastNeuralNetworkDecision(), controlPanelX + 10,
                controlPanelY + 40);
        g2.drawString("ACTUAL DECISION: " + GameLogger.getLastActualDecision(), controlPanelX + 10, controlPanelY + 60);
        g2.drawString("FLIGHT TIME: " + (int) (time * (TIMER_INTERVAL / 1000.0d)) + "s", controlPanelX + 10,
                controlPanelY + 80);

        if (autopilotManager != null && autopilotManager.isRetraining()) {
            g2.setColor(Color.ORANGE);
            g2.drawString("RETRAINING IN PROGRESS...", controlPanelX + 10, controlPanelY + 100);
            g2.setColor(Color.WHITE);
        }
        
        // Add helpful F-key hints
        g2.setColor(new Color(63, 127, 255));
        g2.drawString("F1: Help | F2: Monitor | F3: Teacher | F4: Analyzer", controlPanelX + 10, controlPanelY + 130);

        if (showVisualizations && showConfidenceGraph) {
            drawConfidenceGraph(g2, controlPanelX + 10, controlPanelY + 160);
        }
    }

    /**
     * Draws a confidence graph showing neural network prediction confidence.
     * 
     * @param g2 The Graphics2D object to draw with
     * @param x The x-coordinate to start drawing at
     * @param y The y-coordinate to start drawing at
     */
    private void drawConfidenceGraph(Graphics2D g2, int x, int y) {
        // Get confidence history from the game logger
    	List<Double> confidenceHistory = Arrays.stream(GameLogger.getLastConfidenceScores()).boxed().collect(Collectors.toList());
        if (confidenceHistory == null || confidenceHistory.isEmpty()) {
            g2.setColor(Color.GRAY);
            g2.drawString("No confidence data available", x, y + 10);
            return;
        }
        
        // Graph dimensions
        int graphWidth = 280;
        int graphHeight = 100;
        
        // Draw graph background
        g2.setColor(new Color(0, 0, 0, 120));
        g2.fillRect(x, y, graphWidth, graphHeight);
        
        // Draw graph title
        g2.setColor(Color.WHITE);
        g2.drawString("Neural Network Confidence", x + 5, y + 15);
        
        // Draw axes
        g2.drawLine(x + 5, y + graphHeight - 5, x + graphWidth - 5, y + graphHeight - 5); // X-axis
        g2.drawLine(x + 5, y + 20, x + 5, y + graphHeight - 5); // Y-axis
        
        // Label axes
        g2.setFont(new Font("Monospaced", Font.PLAIN, 10));
        g2.drawString("Time", x + graphWidth/2, y + graphHeight - 5 + 12);
        g2.drawString("100%", x - 5, y + 20);
        g2.drawString("0%", x - 5, y + graphHeight - 5);
        
        // Limit the number of points to display
        int maxPoints = 50;
        int startIdx = Math.max(0, confidenceHistory.size() - maxPoints);
        List<Double> displayedHistory = confidenceHistory.subList(startIdx, confidenceHistory.size());
        
        // Draw data points
        if (displayedHistory.size() > 1) {
            int plotWidth = graphWidth - 15;
            double xStep = (double) plotWidth / (displayedHistory.size() - 1);
            
            // Draw danger threshold (60% confidence)
            g2.setColor(Color.RED);
            int thresholdY = y + 20 + (int)((1 - 0.6) * (graphHeight - 25));
            g2.drawLine(x + 5, thresholdY, x + graphWidth - 5, thresholdY);
            g2.drawString("60%", x - 5, thresholdY);
            
            // Draw confidence line
            g2.setColor(Color.CYAN);
            for (int i = 0; i < displayedHistory.size() - 1; i++) {
                int x1 = x + 5 + (int)(i * xStep);
                int y1 = y + 20 + (int)((1 - displayedHistory.get(i)) * (graphHeight - 25));
                int x2 = x + 5 + (int)((i + 1) * xStep);
                int y2 = y + 20 + (int)((1 - displayedHistory.get(i + 1)) * (graphHeight - 25));
                
                g2.drawLine(x1, y1, x2, y2);
                
                // Highlight low confidence points
                if (displayedHistory.get(i) < 0.6) {
                    g2.setColor(Color.RED);
                    g2.fillOval(x1 - 2, y1 - 2, 4, 4);
                    g2.setColor(Color.CYAN);
                }
            }
            
            // Draw the last point
            int lastIndex = displayedHistory.size() - 1;
            int lastX = x + 5 + (int)(lastIndex * xStep);
            int lastY = y + 20 + (int)((1 - displayedHistory.get(lastIndex)) * (graphHeight - 25));
            if (displayedHistory.get(lastIndex) < 0.6) {
                g2.setColor(Color.RED);
                g2.fillOval(lastX - 2, lastY - 2, 4, 4);
            } else {
                g2.setColor(Color.WHITE);
                g2.fillOval(lastX - 2, lastY - 2, 4, 4);
            }
        }
    }

	public void actionPerformed(ActionEvent e) {
        time++;
        if (autopilotManager != null) {
            autopilotManager.updateFlightTime(time);
        }
        repaint();
        index++;
        index = (index == MODEL_WIDTH) ? 0 : index;
        generateNext();
        if (autopilotManager != null) {
            autoMove();
            
            // Update UI with current game state
            UIManager.getInstance().updateGameState(model, playerRow);
        }
    }

	/**
	 * Handles automated player movement based on autopilot decisions.
	 * Called each game tick when autopilot is enabled.
	 */
	private void autoMove() {
	    if (autopilotManager == null) {
	        return;
	    }
	    
	    // Get action from the autopilot manager
	    int step = autopilotManager.determineAction(model, playerRow);
	    
	    // Apply the movement
	    move(step);
	}
	
	/**
	 * Moves the player based on the given step direction.
	 * 
	 * @param step The direction to move (-1 for up, 0 for stay, 1 for down)
	 */
	public void move(int step) {
	    // Boundary check
	    if ((playerRow == MIN_TOP && step == -1) || (playerRow == MIN_BOTTOM - 1 && step == 1)) {
	        return;
	    }
	    
	    // Move the player
	    playerRow += step;
	    
	    // Check for collision after move
	    if (model.get(PLAYER_COLUMN)[playerRow] != 0) {
	        timer.stop(); // Crash...
	    }
	}

	private void generateNext() {
        byte[] next = model.pollFirst();
        model.addLast(next);
        Arrays.fill(next, ONE_SET);
        int minspace = 4;
        prevTop += current().nextBoolean() ? 1 : -1;
        prevBot += current().nextBoolean() ? 1 : -1;
        prevTop = max(MIN_TOP, min(prevTop, prevBot - minspace));
        prevBot = min(MIN_BOTTOM, max(prevBot, prevTop + minspace));
        Arrays.fill(next, prevTop, prevBot, ZERO_SET);
    }

	public void reset() {
	    // Check if this is a successful flight (timer was running) and autopilot is active
	    boolean wasSuccessfulFlight = timer.isRunning() && autopilotManager != null && autopilotManager.isRecording();
	    boolean shouldConsiderRetraining = false;
	    
	    // Store current mode and flight time before stopping recording
	    AutopilotSystem.Mode currentMode = null;
	    long flightDuration = 0;
	    
	    if (autopilotManager != null) {
	        currentMode = autopilotManager.getMode();
	        flightDuration = time;
	        
	        // Stop recording if needed
	        if (autopilotManager.isRecording()) {
	            autopilotManager.stopRecording(false); // false = successful completion
	            shouldConsiderRetraining = wasSuccessfulFlight;
	        }
	    }
	    
	    // Reset game state
	    model.forEach(n -> Arrays.fill(n, ZERO_SET));
	    playerRow = 11;
	    time = 0;
	    gameOverProcessed = false;
	    timer.restart();
	    
	    // Consider retraining for successful flights
	    if (shouldConsiderRetraining && currentMode != null && currentMode != AutopilotSystem.Mode.NEURAL_NETWORK) {
	        // Define minimum flight time for training (30 seconds)
	        long MINIMUM_FLIGHT_TIME_FOR_TRAINING = 300; // 30 seconds at 100ms intervals
	        
	        if (flightDuration >= MINIMUM_FLIGHT_TIME_FOR_TRAINING) {
	            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
	                    "Successful flight completed. Initiating retraining...");
	            autopilotManager.retrainModel();
	        } else {
	            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
	                    "Flight too short for training: " + flightDuration + " < " + MINIMUM_FLIGHT_TIME_FOR_TRAINING);
	        }
	    }
	    
	    // Restart recording
	    if (autopilotManager != null && !autopilotManager.isRecording()) {
	        autopilotManager.startRecording();
	    }
	    
	    // Update UI state
	    UIManager.getInstance().updateGameState(model, playerRow);
	}

    public void toggleDebugInfo() {
        showDebugInfo = !showDebugInfo;
    }

    public void toggleConfidenceGraph() {
        showConfidenceGraph = !showConfidenceGraph;
    }

    public void toggleVisualizations() {
        showVisualizations = !showVisualizations;
    }
    
}