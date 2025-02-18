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
import ie.atu.sw.autopilot.IAutopilotController;
import ie.atu.sw.autopilot.NeuralNetworkAutopilot;
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
     * The game grid is implemented as a linked list of MODEL_WIDTH columns,
     * where each column is represented by a byte array of size MODEL_HEIGHT.
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
    private final List<TrainingSample> trainingData = new ArrayList<>();
    private static final ExecutorService dataWriterExecutor = Executors.newSingleThreadExecutor();

    // Flight data
    private boolean terminalFlag = false;
    private int lastMovement = 0; // -1 for up, 0 for straight, 1 for down
    private double bestTime = 0;
    private boolean goodFlag = false; // Indicates a "good" flight.
    private double currentFlightTime = 0;
    private static final double MAX_GOOD_FLIGHT_TIME = 30.0; // Flight time threshold in seconds

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
        g2.drawString("Time: " + flightTimeSeconds + "s", 1 * SCALING_FACTOR + 10,
                (15 * SCALING_FACTOR) + (2 * SCALING_FACTOR));

        // Draw Game Over screen.
        if (!timer.isRunning() && gameOver) {
            g2.setFont(gameOverFont);
            g2.setColor(Color.RED);
            g2.drawString("Game Over!", (MODEL_WIDTH / 5) * SCALING_FACTOR, (MODEL_HEIGHT / 2) * SCALING_FACTOR);
        }
    }

    /**
     * Moves the plane up or down.
     * 
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

        // Every 3 ticks, record training sample.
        if (time % 3 == 0) {
            double[] sample = sampleHorizonWithMovementAndPosition();
            trainingData.add(new TrainingSample(sample, lastMovement));

            // Write sample to CSV asynchronously every 10 ticks.
            if (time % 10 == 0) {
                dataWriterExecutor.submit(() -> writeRowToFile("training_data.csv", sample, lastMovement, goodFlag));
            }
        }
    }

    /**
     * Writes a single row to a CSV file. Each row contains the feature vector,
     * the last movement, and the good flight flag.
     *
     * @param fileName     The file to which the row is written.
     * @param sample       The feature vector.
     * @param lastMovement The last movement (-1, 0, or 1).
     * @param goodFlag     True if the flight is considered good.
     */
    private static synchronized void writeRowToFile(String fileName, double[] sample, int lastMovement, boolean goodFlag) {
        try (FileWriter fw = new FileWriter(fileName, true);
             BufferedWriter bw = new BufferedWriter(fw);
             PrintWriter out = new PrintWriter(bw)) {

            StringBuilder sb = new StringBuilder();
            // Append each feature.
            for (double feature : sample) {
                sb.append(feature).append(",");
            }
            sb.append(lastMovement).append(",");
            sb.append(goodFlag ? 1 : 0);

            out.println(sb.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Updates the cave by moving the oldest column to the tail and generating new
     * obstacles.
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
     *
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
     * Samples the horizon (columns ahead of the player) and appends extra
     * features.
     *
     * Feature vector:
     * - Obstacle state for columns ahead of the player.
     * - Last movement.
     * - Normalized player row position.
     * - Terminal flag.
     * - Good flight flag.
     *
     * @return The feature vector.
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
        features[index]   = goodFlag ? 1.0 : 0.0;

        return features;
    }

    /**
     * Ends the game when the plane crashes or goes out-of-bounds.
     */
    public void end() {
        timer.stop();
        currentFlightTime = time * (TIMER_INTERVAL / 1000.0);
        terminalFlag = true;

        // Update good flight flag if flight time exceeds threshold.
        if (currentFlightTime > MAX_GOOD_FLIGHT_TIME && currentFlightTime > bestTime) {
            bestTime = currentFlightTime;
            goodFlag = true;
        } else {
            goodFlag = false;
        }

        // Reset immediately if flight was very short.
        if (currentFlightTime <= 10.0) {
            reset();
        } else {
            gameOver = true;
            repaint();
        }
    }

    /**
     * Resets the game. If autopilot is enabled and training data exists, trains the
     * network before clearing the data.
     */
    public void reset() {
        if (autoMode && !trainingData.isEmpty()) {
            autopilot.trainNetwork(trainingData, 5000);
            trainingData.clear();
        }
        terminalFlag = false;
        model.forEach(column -> Arrays.fill(column, ZERO_SET));
        playerRow = 11;
        time = 0;
        gameOver = false;
        timer.restart();
    }
}
