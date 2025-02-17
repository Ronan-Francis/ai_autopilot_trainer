package ie.atu.sw;

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
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.swing.JPanel;
import javax.swing.Timer;

import ie.atu.sw.trainer.IAutopilotController;
import ie.atu.sw.trainer.NeuralNetworkAutopilot;
import ie.atu.sw.trainer.TrainingSample;

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

	// The player and a sprite for an exploding plane
	private Sprite sprite;
	private Sprite dyingSprite;

	// A flag to indicate if the game is over and waiting for user input
	private boolean gameOver = false;

	private boolean auto;
	private IAutopilotController autopilot;
	private final List<TrainingSample> trainingData = new ArrayList<>();
	private static final ExecutorService dataWriterExecutor = Executors.newSingleThreadExecutor();
	private boolean terminalFlag = false;
	private int lastMovement = 0; // -1 for up, 0 for straight, 1 for down
	private double bestTime = 0; // Track the best (longest) flight time recorded.
	private boolean goodFlag = false;// Flag that indicates a good flight.
	private double currentFlightTime = 0;
	// Set a threshold value that, if exceeded, is considered a good flight.
	private static final double MAX_GOOD_FLIGHT_TIME = 30.0; // For example, 20 seconds.

	public GameView(boolean auto) throws Exception {
		this.auto = auto; // Use the autopilot
		setBackground(Color.LIGHT_GRAY);
		setDoubleBuffered(true);

		// Creates a viewing area of 900 x 600 pixels
		dim = new Dimension(MODEL_WIDTH * SCALING_FACTOR, MODEL_HEIGHT * SCALING_FACTOR);
		super.setPreferredSize(dim);
		super.setMinimumSize(dim);
		super.setMaximumSize(dim);

		initModel();
	    // Input size = (columns ahead of player * MODEL_HEIGHT) + 1 (lastMovement) + 1 (planeRow) 
	    // + 1 (terminal flag) + 1 (good flag) = total extra of 4.
	    if (this.auto) {
	        int inputSize = (MODEL_WIDTH - (PLAYER_COLUMN + 1)) * MODEL_HEIGHT + 4;
	        System.out.println(inputSize);
	        autopilot = new NeuralNetworkAutopilot(inputSize);
	    }

		timer = new Timer(TIMER_INTERVAL, this); // Timer calls actionPerformed() every second
		timer.start();

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

		g2.setColor(Color.WHITE);
		g2.fillRect(0, 0, dim.width, dim.height);

		int x1 = 0, y1 = 0;
		for (int x = 0; x < MODEL_WIDTH; x++) {
			for (int y = 0; y < MODEL_HEIGHT; y++) {
				x1 = x * SCALING_FACTOR;
				y1 = y * SCALING_FACTOR;

				if (model.get(x)[y] != 0) {
					if (y == playerRow && x == PLAYER_COLUMN) {
						end(); // Crash...
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

		if (!timer.isRunning() && gameOver) {
			g2.setFont(over);
			g2.setColor(Color.RED);
			g2.drawString("Game Over!", MODEL_WIDTH / 5 * SCALING_FACTOR, MODEL_HEIGHT / 2 * SCALING_FACTOR);
		}
	}

	// Move the plane up or down
	public void move(int step) {
		playerRow += step;
		lastMovement = step; // Keep track of this move

		// If the plane goes below 0 or above MODEL_HEIGHT - 1, crash the game
		if (playerRow < 0 || playerRow >= MODEL_HEIGHT) {
			end(); // Game Over
		}
	}

	/*
	 * ---------- AUTOPILOT! ---------- The following implementation randomly picks
	 * a -1, 0, 1 to control the plane. You should plug the trained neural network
	 * in here. This method is called by the timer every TIMER_INTERVAL units of
	 * time from actionPerformed(). There are other ways of wiring your neural
	 * network into the application, but this way might be the easiest.
	 * 
	 */
	private void autoMove() {
		if (autopilot != null) {
			double[] state = sample();
			move(autopilot.getMovement(state));
		} else {
			move(current().nextInt(-1, 2));
		}
	}

	// Called every second by the timer
	public void actionPerformed(ActionEvent e) {
		// Increment time & repaint
		time++;
		this.repaint();

		// Generate cave updates
		index++;
		if (index == MODEL_WIDTH) {
			index = 0;
		}
		generateNext();

		// If autopilot is enabled, let it choose the move
		if (auto) {
			autoMove();
		}

		if (time % 3 == 0) {
			double[] sample = sampleHorizonWithMovementAndPosition();
			trainingData.add(new TrainingSample(sample, lastMovement));

			// Every 10 ticks, write the sample to a CSV file asynchronously.
			if (time % 10 == 0) {
				dataWriterExecutor.submit(() -> writeRowToFile("training_data.csv", sample, lastMovement));
			}
		}
	}

	private void writeRowToFile(String filename, double[] features, double label) {
		try (FileWriter fw = new FileWriter(filename, true);
				BufferedWriter bw = new BufferedWriter(fw);
				PrintWriter out = new PrintWriter(bw)) {

			// Print each feature separated by commas
			for (int i = 0; i < features.length; i++) {
				out.print(features[i]);
				if (i < features.length - 1) {
					out.print(",");
				}
			}
			// Finally print the label, then newline
			out.print("," + label);
			out.println();

		} catch (IOException e) {
			e.printStackTrace();
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

	/*
	 * Use this method to get a snapshot of the 30x20 matrix of values that make up
	 * the game grid. The grid is flatmapped into a single dimension double array...
	 * (somewhat) ready to be used by a neural net. You can experiment around with
	 * how much of this you actually will need. The plane is always somehere in
	 * column PLAYER_COLUMN and you probably do not need any of the columns behind
	 * this. You can consider all of the columns ahead of PLAYER_COLUMN as your
	 * horizon and this value can be reduced to save space and time if needed, e.g.
	 * just look 1, 2 or 3 columns ahead.
	 * 
	 * You may also want to track the last player movement, i.e. up, down or no
	 * change. Depending on how you design your neural network, you may also want to
	 * label the data as either okay or dead. Alternatively, the label might be the
	 * movement (up, down or straight).
	 * 
	 */
	public double[] sample() {
		var vector = new double[MODEL_WIDTH * MODEL_HEIGHT];
		var index = 0;

		for (byte[] bm : model) {
			for (byte b : bm) {
				vector[index] = b;
				index++;
			}
		}
		return vector;
	}

	public double[] sampleHorizonWithMovementAndPosition() {
	    int horizonStart = PLAYER_COLUMN + 1;
	    int horizonColumns = MODEL_WIDTH - horizonStart;

	    // Now, feature vector size = horizonColumns * MODEL_HEIGHT + 4.
	    double[] features = new double[horizonColumns * MODEL_HEIGHT + 4];
	    int index = 0;

	    // 1) Fill horizon columns.
	    for (int x = horizonStart; x < MODEL_WIDTH; x++) {
	        byte[] col = model.get(x);
	        for (int y = 0; y < MODEL_HEIGHT; y++) {
	            features[index++] = col[y];
	        }
	    }

	    // 2) Append lastMovement.
	    features[index++] = lastMovement;

	    // 3) Append plane’s normalized row position.
	    features[index++] = (double) playerRow / MODEL_HEIGHT;
	    
	    // 4) Append terminal flag.
	    // For a terminal state, you may already have set a flag elsewhere.
	    // For now, we assume end() sets terminalFlag (see later) – here we read it.
	    features[index++] = terminalFlag ? 1.0 : 0.0;
	    
	    // 5) Append good flight flag.
	    features[index] = goodFlag ? 1.0 : 0.0;

	    return features;
	}


	/**
	 * Called when the game ends (e.g. crash/out-of-bounds).
	 */
	public void end() {
	    timer.stop(); // Stop updating the game

	    // Calculate how many seconds have passed.
	    currentFlightTime = time * (TIMER_INTERVAL / 1000.0);
	    
	    // Mark the current state as terminal.
	    terminalFlag = true;
	    
	    // If this flight exceeds our maximum "good" flight time threshold,
	    // update bestTime and mark the flight as good.
	    if (currentFlightTime > MAX_GOOD_FLIGHT_TIME && currentFlightTime > bestTime) {
	        bestTime = currentFlightTime;
	        goodFlag = true;
	    } else {
	        goodFlag = false;
	    }

	    if (currentFlightTime <= 10.0) {
	        reset();
	    } else {
	        gameOver = true;
	        repaint();
	    }
	}

	/*
	 * Resets and restarts the game when the "S" key is pressed
	 */
	public void reset() {
		if (auto && !trainingData.isEmpty()) {
			autopilot.trainNetwork(trainingData, 2500);
			trainingData.clear();
		}

		terminalFlag = false; // Reset the terminal flag.
		model.stream().forEach(n -> Arrays.fill(n, 0, n.length, ZERO_SET));
		playerRow = 11; // Centre the plane
		time = 0; // Reset the clock
		timer.restart(); // Start the animation
	}
}