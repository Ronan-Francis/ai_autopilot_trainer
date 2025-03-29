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

import javax.swing.JPanel;
import javax.swing.Timer;

import ie.atu.sw.network.FeatureExtractor;
import ie.atu.sw.network.FuzzyController;
import ie.atu.sw.network.FuzzyVisualizer;
import ie.atu.sw.network.GameLogger;
import ie.atu.sw.network.HybridController;
import ie.atu.sw.network.NeuralNetworkController;

public class GameView extends JPanel implements ActionListener {
	// Some constants
	private static final long serialVersionUID = 1L;
	private static final int MODEL_WIDTH = 30;
	private static final int MODEL_HEIGHT = 20;
	private static final int SCALING_FACTOR = 30;

	private static final int MIN_TOP = 2;
	private static final int MIN_BOTTOM = 18;
	private static final int PLAYER_COLUMN = 15;
	private static final int TIMER_INTERVAL = 100;

	private static final byte ONE_SET = 1;
	private static final byte ZERO_SET = 0;

	/*
	 * The 30x20 game grid is implemented using a linked list of 30 elements, where
	 * each element contains a byte[] of size 20.
	 */
	private LinkedList<byte[]> model = new LinkedList<>();

	// These two variables are used by the cavern generator.
	private int prevTop = MIN_TOP;
	private int prevBot = MIN_BOTTOM;

	// Once the timer stops, the game is over
	private Timer timer;
	private long time;

	private int playerRow = 11;
	private int index = MODEL_WIDTH - 1; // Start generating at the end
	private Dimension dim;

	// Some fonts for the UI display
	private Font font = new Font("Dialog", Font.BOLD, 50);
	private Font over = new Font("Dialog", Font.BOLD, 100);
	private Font debugFont = new Font("Monospaced", Font.PLAIN, 14);

	// The player and a sprite for an exploding plane
	private Sprite sprite;
	private Sprite dyingSprite;

	private boolean auto;
	private boolean showVisualizations = false; // Initially off

	// Neural network components
	private FeatureExtractor featureExtractor;
	private NeuralNetworkController nnController;
	private FuzzyController fuzzyController;
	private final Object trainingLock = new Object();
	
	// Debug display options
	private boolean showDebugInfo = true;
	private boolean showConfidenceGraph = true;
	private ie.atu.sw.network.FuzzyVisualizer fuzzyVisualizer;
	private HybridController hybridController;// Use both NN and fuzzy logic together

	public GameView(boolean auto) throws Exception {
		this.auto = auto; // Use the autopilot
		setBackground(Color.LIGHT_GRAY);
		setDoubleBuffered(true);

		// Creates a viewing area of 900 x 600 pixels
		dim = new Dimension(MODEL_WIDTH * SCALING_FACTOR, MODEL_HEIGHT * SCALING_FACTOR);
		super.setPreferredSize(dim);
		super.setMinimumSize(dim);
		super.setMaximumSize(dim);

		// Initialize logging
		GameLogger.initialize();
		GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL, 
				"Game initialized with auto mode: " + auto);

		initModel();

		// Initialize AI components
		initAI();

		timer = new Timer(TIMER_INTERVAL, this); // Timer calls actionPerformed() every second
		timer.start();
	}

	private void initAI() {
	    // Create feature extractors (both original and simplified)
	    featureExtractor = new FeatureExtractor();

	    // Create neural network controllers
	    nnController = new NeuralNetworkController(featureExtractor);

	    // Create fuzzy controllers
	    fuzzyController = new FuzzyController(); 
	    fuzzyVisualizer = new FuzzyVisualizer(fuzzyController); 
	    
	    // Create the hybrid controller
	    hybridController = new HybridController(nnController, fuzzyController);
	    
	    // Create fuzzy visualizer if debug is enabled
	    //fuzzyVisualizer = new FuzzyVisualizer(impFuzzyController);

	    // Train the networks if in auto mode (assuming training data exists)
	    if (auto) {
	        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
	                "Auto mode detected. Attempting to load neural network models...");

	        // Try to load both models, or train if not available
	        boolean originalLoaded = nnController.loadModel();
	        
	        if (!originalLoaded) {
	            GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.NEURAL_NETWORK,
	                    "One or more models not found or loading failed. Attempting to train...");
	            synchronized (trainingLock) {
	                if (!originalLoaded) nnController.trainNetwork();
	            }
	        }
	    }
	}


	// Build our game grid
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

	// Called every second by actionPerformed(). Paint methods are usually ugly.
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		var g2 = (Graphics2D) g;

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

		/*
		 * Not pretty, but good enough for this project... The compiler will tidy up and
		 * optimise all of the arithmetics with constants below.
		 */
		g2.setFont(font);
		g2.setColor(Color.RED);
		g2.fillRect(1 * SCALING_FACTOR, 15 * SCALING_FACTOR, 400, 3 * SCALING_FACTOR);
		g2.setColor(Color.WHITE);
		g2.drawString("Time: " + (int) (time * (TIMER_INTERVAL / 1000.0d)) + "s", 1 * SCALING_FACTOR + 10,
				(15 * SCALING_FACTOR) + (2 * SCALING_FACTOR));
		
		// Draw debug info if enabled
		if (showDebugInfo && auto) {
			drawDebugInfo(g2);
		}

		if (!timer.isRunning()) {
		    g2.setFont(over);
		    g2.setColor(Color.RED);
		    g2.drawString("Game Over!", MODEL_WIDTH / 5 * SCALING_FACTOR, MODEL_HEIGHT / 2 * SCALING_FACTOR);
		    
		    // When game is over, discard recent samples and train the neural network 
		    if (!auto && featureExtractor != null) {
		        // Discard samples just before crash
		        featureExtractor.discardBufferedSamples();
		        
		        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
		                "Game over. Training neural network with " + 
		                featureExtractor.getSampleCount() + " collected samples...");
		        new Thread(() -> {
		            synchronized(trainingLock) {
		                nnController.trainNetwork();
		            }
		        }).start();
		    }
		}
	}
	
	/**
	 * Draw debug information on the screen.
	 * 
	 * @param g2 Graphics context
	 */
	private void drawDebugInfo(Graphics2D g2) {
	    g2.setFont(debugFont);
	    
	    // Draw decision information
	    int startX = 5 * SCALING_FACTOR;
	    int startY = 1 * SCALING_FACTOR;
	    
	    g2.setColor(new Color(0, 0, 0, 180)); // Semi-transparent black background
	    g2.fillRect(startX, startY, 19 * SCALING_FACTOR, 5 * SCALING_FACTOR);
	    
	    g2.setColor(Color.WHITE);
	    g2.drawString("ORIGINAL NN: " + GameLogger.getLastNeuralNetworkDecision(), 
	            startX + 10, startY + 20);
	    g2.drawString("FUZZY CONTROLLER: " + GameLogger.getLastFuzzyControllerDecision(), 
	            startX + 10, startY + 60);
	    g2.drawString("ACTUAL DECISION: " + GameLogger.getLastActualDecision(), 
	            startX + 10, startY + 80);
	    g2.drawString("HYBRID MODE: " + (hybridController.isHybridModeEnabled() ? "ENABLED" : "DISABLED"), 
	            startX + 10, startY + 100);
	    
	    // Only draw visualizations if the flag is enabled
	    if (showVisualizations) {
	        // Draw confidence bar graph if enabled
	        if (showConfidenceGraph) {
	            drawConfidenceGraph(g2, startX + 10, startY + 120);
	        }
	        
	        // Draw fuzzy controller visualization
	        if (fuzzyVisualizer != null) {
	            fuzzyVisualizer.draw(g2, model, playerRow, startX + 500, startY + 220);
	        }
	    }
	}
	/**
	 * Draw a bar graph of the neural network confidence scores.
	 * 
	 * @param g2 Graphics context
	 * @param x X position
	 * @param y Y position
	 */
	private void drawConfidenceGraph(Graphics2D g2, int x, int y) {
		double[] scores = GameLogger.getLastConfidenceScores();
		
		if (scores == null) {
			return;
		}
		
		int barWidth = 40;
		int barMaxHeight = 80;
		int spacing = 20;
		
		g2.setColor(Color.WHITE);
		g2.drawString("CONFIDENCE SCORES:", x, y);
		
		y += 20;
		
		String[] actions = {"UP", "STAY", "DOWN"};
		Color[] colors = {Color.BLUE, Color.GREEN, Color.RED};
		
		for (int i = 0; i < scores.length; i++) {
			int barHeight = (int)(scores[i] * barMaxHeight);
			
			g2.setColor(colors[i]);
			g2.fillRect(x + (barWidth + spacing) * i, 
					y + barMaxHeight - barHeight, 
					barWidth, barHeight);
			
			g2.setColor(Color.WHITE);
			g2.drawRect(x + (barWidth + spacing) * i, 
					y, barWidth, barMaxHeight);
			
			String scoreText = String.format("%.1f%%", scores[i] * 100);
			g2.drawString(actions[i], 
					x + (barWidth + spacing) * i + barWidth/2 - 10, 
					y + barMaxHeight + 15);
			g2.drawString(scoreText, 
					x + (barWidth + spacing) * i + barWidth/2 - 15, 
					y + barMaxHeight + 30);
		}
	}

	// Move the plane up or down
	public void move(int step) {
	    // Ensure we stay within bounds
	    int newPlayerRow = playerRow + step;
	    if (newPlayerRow < 0) {
	        step = -playerRow; // Stay at top
	    } else if (newPlayerRow >= MODEL_HEIGHT) {
	        step = MODEL_HEIGHT - 1 - playerRow; // Stay at bottom
	    }
	    
	    // In manual mode, record the training sample and check for danger
	    if (!auto && featureExtractor != null && timer.isRunning()) {
	        // Extract features from current state
	        double[] features = featureExtractor.extractFeatures(model, playerRow);
	        
	        // Check if we're in danger using the fuzzy controller
	        boolean inDanger = false;
	        int fuzzyAction = 0;
	        
	        if (fuzzyController != null) {
	            // Check for danger
	            inDanger = fuzzyController.getLastDangerState();
	            
	            // If in danger, get fuzzy controller recommendation
	            if (inDanger) {
	                fuzzyAction = fuzzyController.determineAction(model, playerRow);
	                GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FUZZY_CONTROLLER,
	                    "MANUAL MODE: Danger detected! Fuzzy suggests: " + actionToString(fuzzyAction));
	                
	                // We don't force the fuzzy action, but we record it for training
	                GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GAME_STATE,
	                    "MANUAL: Fuzzy suggestion " + actionToString(fuzzyAction) + 
	                    ", Player chose " + actionToString(step));
	            }
	        }
	        
	        // Map the manual action to a label
	        int action = featureExtractor.mapActionToLabel(step);
	        
	        // Record the sample with the manual action
	        featureExtractor.recordSample(features, action);
	        
	        // Set last action
	        featureExtractor.setLastAction(action);
	        
	        // Log the manual action
	        GameLogger.logFinalAction(step);
	    }
	    
	    // Actually move the player
	    playerRow += step;
	}

	/**
	 * Helper method to convert action to string
	 */
	private String actionToString(int action) {
	    switch (action) {
	        case -1: return "UP";
	        case 0: return "STAY";
	        case 1: return "DOWN";
	        default: return "UNKNOWN";
	    }
	}
	/*
	 * ---------- AUTOPILOT! ---------- The following implementation uses the neural
	 * network to control the plane.
	 * 
	 */
	// Fixed autoMove method with improved logic

	/**
	 * Controls the autopilot movement using neural network and improved fuzzy logic.
	 * This updated version uses adaptive thresholds and better path planning.
	 */
	private void autoMove() {
	    if (nnController == null) {
	        move(current().nextInt(-1, 2)); // fallback
	        return;
	    }

	    // Use the hybrid controller to determine the action
	    int action = hybridController.determineAction(model, playerRow);

	    // Constrain movement within game bounds
	    int newRow = playerRow + action;
	    if (newRow < 0) {
	        action = -playerRow;
	    } else if (newRow >= MODEL_HEIGHT) {
	        action = MODEL_HEIGHT - 1 - playerRow;
	    }

	    // Move the player
	    move(action);
	}
	
	
	public void actionPerformed(ActionEvent e) {
	    time++; // Update our timer		
	    GameLogger.logGameState(playerRow, time);// Log game state		
	    this.repaint(); // Repaint the cavern
	    
	    // Update the next index to generate
	    index++;
	    index = (index == MODEL_WIDTH) ? 0 : index;

	    generateNext(); // Generate the next part of the cave
	    
	    // In auto mode, use the AI to control the plane
	    if (auto) {
	        autoMove();
	    } 
	    // In manual mode, still evaluate danger with fuzzy controller (but don't move)
	    else if (fuzzyController != null) {
	        // Just determine the action to update the danger state
	        fuzzyController.determineAction(model, playerRow);
	    }

	    // Record training data in manual mode on every frame
	    if (!auto && featureExtractor != null) {
	        double[] features = featureExtractor.extractFeatures(model, playerRow);
	        int action = featureExtractor.getLastAction();
	        
	        // We already record during moves, but this captures state even when not moving
	        if (time % 3 == 0) { // Reduce frequency a bit to avoid too many "stay" samples
	            featureExtractor.recordSample(features, action);
	        }
	        
	        // Commit samples periodically during successful flight
	        if (time > 0 && time % 30 == 0) {
	            commitSamplesIfSuccessful();
	        }
	    }
	}
	
	private void commitSamplesIfSuccessful() {
	    // Consider flight successful if it has lasted for at least a few seconds
	    if (!auto && featureExtractor != null && time > 0 && time % 30 == 0) {
	        featureExtractor.commitAllSamples();
	        GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FEATURE_EXTRACTOR,
	                "Committed samples after " + (int)(time * (TIMER_INTERVAL / 1000.0d)) + "s of successful flight");
	    }
	}

	// Add method to clear training data
	public void clearTrainingData() {
		if (featureExtractor != null) {
			GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.FEATURE_EXTRACTOR,
					"Clearing training data...");
			featureExtractor.clearTrainingData();
		}
	}

	/*
	 * Generate the next layer of the cavern. Use the linked list to move the
	 * current head element to the tail and then randomly decide whether to increase
	 * or decrease the cavern.
	 */
	private void generateNext() {
		var next = model.pollFirst();
		model.addLast(next); // Move the head to the tail
		Arrays.fill(next, ONE_SET); // Fill everything in

		// Flip a coin to determine if we could grow or shrink the cave
		var minspace = 4; // Smaller values will create a cave with smaller spaces
		prevTop += current().nextBoolean() ? 1 : -1;
		prevBot += current().nextBoolean() ? 1 : -1;
		prevTop = max(MIN_TOP, min(prevTop, prevBot - minspace));
		prevBot = min(MIN_BOTTOM, max(prevBot, prevTop + minspace));

		// Fill in the array with the carved area
		Arrays.fill(next, prevTop, prevBot, ZERO_SET);
	}

	/**
	 * Returns the neural network controller for external use.
	 * 
	 * @return The neural network controller or null if not initialized
	 */
	public NeuralNetworkController getNeuralNetworkController() {
		return nnController;
	}

	/**
	 * Resets and restarts the game when the "S" key is pressed. Also resets the
	 * player position and game time.
	 */
	public void reset() {
		GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GAME_STATE,
				"Game reset");
				
		model.stream() // Zero out the grid
				.forEach(n -> Arrays.fill(n, 0, n.length, ZERO_SET));
		playerRow = 11; // Centre the plane
		time = 0; // Reset the clock
		timer.restart(); // Start the animation
	}

	/**
	 * Toggles between using just the neural network and the hybrid approach. When
	 * hybrid control is enabled, the fuzzy controller is used as a fallback.
	 */
	public void toggleHybridControl() {
	    hybridController.toggleHybridMode();
	    GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
	            "Hybrid control " + (hybridController.isHybridModeEnabled() ? "enabled" : "disabled"));
	}
	
	/**
	 * Toggles the debug information display.
	 */
	public void toggleDebugInfo() {
		showDebugInfo = !showDebugInfo;
		GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
				"Debug info display " + (showDebugInfo ? "enabled" : "disabled"));
	}
	
	/**
	 * Toggles the confidence graph display.
	 */
	public void toggleConfidenceGraph() {
		showConfidenceGraph = !showConfidenceGraph;
		GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
				"Confidence graph " + (showConfidenceGraph ? "enabled" : "disabled"));
	}
	/**
	 * Toggles the visualization display.
	 */
	public void toggleVisualizations() {
	    showVisualizations = !showVisualizations;
	    GameLogger.log(GameLogger.LogLevel.INFO, GameLogger.LogCategory.GENERAL,
	            "Visualizations " + (showVisualizations ? "enabled" : "disabled"));
	}
}