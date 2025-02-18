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
import java.util.Arrays;
import java.util.LinkedList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import javax.swing.JPanel;
import javax.swing.Timer;

import ie.atu.sw.autopilot.IAutopilotController;
import ie.atu.sw.autopilot.NeuralNetworkAutopilot;
import ie.atu.sw.autopilot.TrainingDataBuffer; // NEW
import ie.atu.sw.autopilot.TrainingSample;

public class GameView extends JPanel implements ActionListener {
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
	 * The game grid is implemented as a linked list of MODEL_WIDTH columns, where
	 * each column is represented by a byte array of size MODEL_HEIGHT.
	 */
	private final LinkedList<byte[]> model = new LinkedList<>();

	// Variables for the cavern generator.
	private int prevTop = MIN_TOP;
	private int prevBot = MIN_BOTTOM;

	private Timer timer;
	private long time;

	private int playerRow = 11;
	private final Dimension dim;

	// Fonts for UI display.
	private final Font timeFont = new Font("Dialog", Font.BOLD, 50);
	private final Font gameOverFont = new Font("Dialog", Font.BOLD, 100);

	// Sprites for the plane and explosion.
	private Sprite sprite;
	private Sprite dyingSprite;

	// Game state flags.
	private boolean gameOver = false;
	private boolean autoMode;
	private IAutopilotController autopilot;

	// Instead of storing TrainingSample lists, store them in the buffers below.
	private final TrainingDataBuffer trainingDataBuffer = new TrainingDataBuffer();   // Good flights
	private final TrainingDataBuffer currentFlightBuffer = new TrainingDataBuffer(); // Current flight

	private static final ExecutorService dataWriterExecutor = Executors.newSingleThreadExecutor();

	// Flight data
	private boolean terminalFlag = false;
	private int lastMovement = 0; // -1 for up, 0 for straight, 1 for down
	private double bestTime = 0;
	private boolean goodFlag = false; // Indicates a "good" flight.
	private double currentFlightTime = 0;
	private static final double TIME_WEIGHT = 10.0; // Reward per second of flight time
	private static final double GOOD_FLIGHT_THRESHOLD = 100.0;

	public GameView(boolean autoMode) throws Exception {
		this.autoMode = autoMode;
		setBackground(Color.LIGHT_GRAY);
		setDoubleBuffered(true);

		// Set panel size
		dim = new Dimension(MODEL_WIDTH * SCALING_FACTOR, MODEL_HEIGHT * SCALING_FACTOR);
		setPreferredSize(dim);
		setMinimumSize(dim);
		setMaximumSize(dim);

		initModel();

		// Input size for the neural network:
		// (columns ahead of player * MODEL_HEIGHT) + 4 extra features.
		if (this.autoMode) {
			int horizonColumns = MODEL_WIDTH - (PLAYER_COLUMN + 1);
			int inputSize = (horizonColumns * MODEL_HEIGHT) + 4;
			System.out.println("Neural Network Input Size: " + inputSize);
			autopilot = new NeuralNetworkAutopilot(inputSize);
		}

		timer = new Timer(TIMER_INTERVAL, this);
		timer.start();
	}

	/**
	 * Initializes the game grid with empty (zero) values.
	 */
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

	@Override
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		Graphics2D g2 = (Graphics2D) g;

		// Draw background.
		g2.setColor(Color.WHITE);
		g2.fillRect(0, 0, dim.width, dim.height);

		// Draw grid and sprites.
		for (int x = 0; x < MODEL_WIDTH; x++) {
			for (int y = 0; y < MODEL_HEIGHT; y++) {
				int x1 = x * SCALING_FACTOR;
				int y1 = y * SCALING_FACTOR;

				// Draw obstacles.
				if (model.get(x)[y] != 0) {
					// If the plane collides with an obstacle, end the game.
					if (y == playerRow && x == PLAYER_COLUMN) {
						end();
					}
					g2.setColor(Color.BLACK);
					g2.fillRect(x1, y1, SCALING_FACTOR, SCALING_FACTOR);
				}

				// Draw the player.
				if (x == PLAYER_COLUMN && y == playerRow) {
					if (timer.isRunning()) {
						g2.drawImage(sprite.getNext(), x1, y1, null);
					} else {
						g2.drawImage(dyingSprite.getNext(), x1, y1, null);
					}
				}
			}
		}

		// Draw UI: flight time display.
		g2.setFont(timeFont);
		g2.setColor(Color.RED);
		g2.fillRect(1 * SCALING_FACTOR, 15 * SCALING_FACTOR, 400, 3 * SCALING_FACTOR);
		g2.setColor(Color.WHITE);
		int flightTimeSeconds = (int) (time * (TIMER_INTERVAL / 1000.0));
		g2.drawString("Time: " + flightTimeSeconds + "s", 
				1 * SCALING_FACTOR + 10, 
				(15 * SCALING_FACTOR) + (2 * SCALING_FACTOR));

		// Draw Game Over screen.
		if (!timer.isRunning() && gameOver) {
			g2.setFont(gameOverFont);
			g2.setColor(Color.RED);
			g2.drawString("Game Over!", 
					(MODEL_WIDTH / 5) * SCALING_FACTOR, 
					(MODEL_HEIGHT / 2) * SCALING_FACTOR);
		}
	}

	/**
	 * Moves the plane up or down.
	 * @param step -1 for up, 0 for straight, 1 for down.
	 */
	public void move(int step) {
		playerRow += step;
		lastMovement = step;

		// Check bounds.
		if (playerRow < 0 || playerRow >= MODEL_HEIGHT) {
			end();
		}
	}

	/**
	 * Invokes autopilot movement if enabled; otherwise, moves randomly.
	 */
	private void autoMove() {
		if (autopilot != null) {
			double[] state = sample();
			move(autopilot.getMovement(state));
		} else {
			move(current().nextInt(-1, 2));
		}
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		time++;
		repaint();

		updateCave();
		if (autoMode) {
			autoMove();
		}

		// Every 3 ticks, record training sample for the current flight.
		if (time % 3 == 0) {
			double[] sample = sampleHorizonWithMovementAndPosition();
			
			//Convert lastMovement to one-hot and store in currentFlightBuffer.
			double[] labelOneHot = toOneHot(lastMovement);
			currentFlightBuffer.addSample(sample, labelOneHot);
			
			// Optionally still write to CSV 
			if (time % 10 == 0) {
				dataWriterExecutor.submit(() -> writeRowToFile("training_data.csv", sample, lastMovement));
			}
		}
	}

	/**
	 * Convert a movement (–1, 0, 1) to a one-hot vector of length 3.
	 * up   (–1) → [1, 0, 0]
	 * stay ( 0) → [0, 1, 0]
	 * down ( 1) → [0, 0, 1]
	 */
	private double[] toOneHot(int movement) {
		double[] labelOneHot = new double[3];
		if (movement == -1) {
			labelOneHot[0] = 1.0;
		} else if (movement == 0) {
			labelOneHot[1] = 1.0;
		} else {
			labelOneHot[2] = 1.0;
		}
		return labelOneHot;
	}

	/**
	 * Writes a single row to a CSV file. Each row contains the feature vector,
	 * the last movement, etc.
	 */
	private static synchronized void writeRowToFile(String fileName, double[] sample, int lastMovement) {
		try (FileWriter fw = new FileWriter(fileName, true);
				BufferedWriter bw = new BufferedWriter(fw);
				PrintWriter out = new PrintWriter(bw)) {

			StringBuilder sb = new StringBuilder();
			// Append each feature.
			for (double feature : sample) {
				sb.append(feature).append(",");
			}
			sb.append(lastMovement);
			out.println(sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Updates the cave by moving the oldest column to the tail and generating new obstacles.
	 */
	private void updateCave() {
		byte[] nextColumn = model.pollFirst();
		model.addLast(nextColumn);
		Arrays.fill(nextColumn, ONE_SET);

		// Determine new cavern boundaries.
		int minSpace = 4; // Minimum gap size.
		prevTop += current().nextBoolean() ? 1 : -1;
		prevBot += current().nextBoolean() ? 1 : -1;
		prevTop = max(MIN_TOP, min(prevTop, prevBot - minSpace));
		prevBot = min(MIN_BOTTOM, max(prevBot, prevTop + minSpace));

		// Carve out the cavern.
		Arrays.fill(nextColumn, prevTop, prevBot, ZERO_SET);
	}

	/**
	 * Samples the entire game grid (flattened) for use by the neural network.
	 * @return A double array representation of the grid.
	 */
	public double[] sample() {
		double[] vector = new double[MODEL_WIDTH * MODEL_HEIGHT];
		int index = 0;
		for (byte[] column : model) {
			for (byte cell : column) {
				vector[index++] = cell;
			}
		}
		return vector;
	}

	/**
	 * Samples the horizon (columns ahead of the player) and appends extra features:
	 *  - obstacle states for columns ahead
	 *  - lastMovement
	 *  - normalized player row
	 *  - terminalFlag
	 *  - goodFlag
	 */
	public double[] sampleHorizonWithMovementAndPosition() {
		int horizonStart = PLAYER_COLUMN + 1;
		int horizonColumns = MODEL_WIDTH - horizonStart;
		int featureVectorSize = (horizonColumns * MODEL_HEIGHT) + 4;
		double[] features = new double[featureVectorSize];
		int index = 0;

		// Append horizon columns.
		for (int x = horizonStart; x < MODEL_WIDTH; x++) {
			byte[] column = model.get(x);
			for (int y = 0; y < MODEL_HEIGHT; y++) {
				features[index++] = column[y];
			}
		}

		// Append extra features.
		features[index++] = lastMovement;
		features[index++] = (double) playerRow / MODEL_HEIGHT;
		features[index++] = terminalFlag ? 1.0 : 0.0;
		features[index]   = goodFlag     ? 1.0 : 0.0;

		return features;
	}

	/**
	 * Ends the game when the plane crashes or goes out-of-bounds.
	 */
	public void end() {
		timer.stop();
		currentFlightTime = time * (TIMER_INTERVAL / 1000.0);
		terminalFlag = true;

		double flightScore = computeFlightScore();
		System.out.println("Flight score: " + flightScore);

		// Update good flight flag if flight time or flightScore exceed threshold.
		if (currentFlightTime > bestTime || flightScore > GOOD_FLIGHT_THRESHOLD) {
			goodFlag = true;
			bestTime = currentFlightTime;
		} else {
			goodFlag = false;
		}

		// If flight qualifies as "good," merge current flight data into the main buffer
		// and then clear the current buffer.
		if (goodFlag) {
			mergeCurrentFlightIntoMain();
		}

		// If flight was extremely short, reset right away.
		if (currentFlightTime <= 10.0) {
			reset();
			mergeCurrentFlightIntoMain();
		} else {
			gameOver = true;
			repaint();
		}
	}

	/**
	 * Merges the samples from currentFlightBuffer into trainingDataBuffer, then clears currentFlightBuffer.
	 */
	private void mergeCurrentFlightIntoMain() {
		double[][] flightFeatures = currentFlightBuffer.getFeaturesArray();
		double[][] flightLabels = currentFlightBuffer.getLabelArray();
		for (int i = 0; i < flightFeatures.length; i++) {
			trainingDataBuffer.addSample(flightFeatures[i], flightLabels[i]);
		}
		currentFlightBuffer.clear();
	}

	/**
	 * Resets the game. If autopilot is enabled and we have training data, train the network.
	 */
	public void reset() {
		if (autoMode && trainingDataBuffer.size() > 0) {
			trainingDataBuffer.saveToCSV("training_data.csv", (int)currentFlightBuffer.getFeaturesArray().length);
			trainAutopilotUsingBuffer();
		}

		terminalFlag = false;
		model.forEach(column -> Arrays.fill(column, ZERO_SET));
		playerRow = 11;
		time = 0;
		gameOver = false;
		timer.restart();
	}

	/**
	 * Uses the stored training samples in trainingDataBuffer to train the autopilot.
	 */
	private void trainAutopilotUsingBuffer() {
		// The autopilot’s trainNetwork(...) method currently expects a List<TrainingSample>.
		// quickly build up a List<TrainingSample> on-the-fly.

		// Convert the buffer to arrays:
		double[][] features = trainingDataBuffer.getFeaturesArray();
		double[][] labels   = trainingDataBuffer.getLabelArray();

		// Convert each row of 'labels' from one-hot to (–1, 0, or 1), then create TrainingSample.
		java.util.List<TrainingSample> samples = new java.util.ArrayList<>();
		for (int i = 0; i < features.length; i++) {
			int m = labelIndexToMovement(labels[i]);
			samples.add(new TrainingSample(features[i], m));
		}

		// Now train:
		autopilot.trainNetwork(samples, 5000);

		// Clear the buffer after a successful train, if desired:
		trainingDataBuffer.clear();
	}

	/**
	 * Helper that maps a one-hot vector back to an integer movement.
	 * [1,0,0] → -1; [0,1,0] → 0; [0,0,1] → 1
	 */
	private int labelIndexToMovement(double[] oneHot) {
		// if oneHot[0] = 1 → -1
		// if oneHot[1] = 1 → 0
		// if oneHot[2] = 1 → 1
		if (oneHot[0] > 0.5) return -1;
		if (oneHot[1] > 0.5) return 0;
		return 1;
	}

	private double computeFlightScore() {
		if (time == 0) {
			return 0;
		}
		return (time * TIME_WEIGHT);
	}
}
