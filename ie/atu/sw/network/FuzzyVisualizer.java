package ie.atu.sw.network;

import java.awt.AlphaComposite;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Composite;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.Stroke;
import java.util.LinkedList;

/**
 * Visualizes the fuzzy controller's decision-making process,
 * showing tunnel centers, target rows, and safety ratings.
 */
public class FuzzyVisualizer {
    // Constants
    private static final int MODEL_HEIGHT = 20;
    private static final int PLAYER_COLUMN = 15;
    private static final int LOOK_AHEAD = 4;
    
    // Visual parameters
    private static final int CELL_SIZE = 20;
    private static final int GRID_WIDTH = 8;
    private static final int GRID_HEIGHT = MODEL_HEIGHT;
    
    private static final Color DANGER_COLOR = new Color(255, 0, 0, 128);
    private static final Color SAFE_COLOR = new Color(0, 255, 0, 128);
    private static final Color TARGET_COLOR = new Color(0, 0, 255, 180);
    private static final Color TUNNEL_COLOR = new Color(255, 255, 0, 180);
    
    private final FuzzyController fuzzyController;
    private Font labelFont = new Font("SansSerif", Font.PLAIN, 10);
    
    /**
     * Creates a new fuzzy visualizer connected to a fuzzy controller.
     * 
     * @param fuzzyController The fuzzy controller to visualize
     */
    public FuzzyVisualizer(FuzzyController fuzzyController) {
        this.fuzzyController = fuzzyController;
    }
    
    /**
     * Draws the fuzzy controller visualization on the game view.
     * 
     * @param g2 Graphics context
     * @param model Game grid model
     * @param playerRow Current player row
     * @param x Starting x coordinate for drawing
     * @param y Starting y coordinate for drawing
     */
    public void draw(Graphics2D g2, LinkedList<byte[]> model, int playerRow, int x, int y) {
        if (fuzzyController == null) return;
        
        // Store original graphics state
        Stroke originalStroke = g2.getStroke();
        Composite originalComposite = g2.getComposite();
        Font originalFont = g2.getFont();
        
        // Draw header
        g2.setColor(Color.WHITE);
        g2.setFont(new Font("SansSerif", Font.BOLD, 12));
        g2.drawString("FUZZY CONTROLLER VISUALIZATION", x, y);
        y += 20;
        
        // Draw visualization grid background
        drawGridBackground(g2, x, y);
        
        // Draw tunnel centers
        int[] tunnelCenters = fuzzyController.getLastTunnelCenters();
        drawTunnelCenters(g2, tunnelCenters, x, y);
        
        // Draw target row
        int targetRow = fuzzyController.getLastTargetRow();
        drawTargetRow(g2, targetRow, x, y);
        
        // Draw danger state indicator
        boolean dangerState = fuzzyController.getLastDangerState();
        drawDangerIndicator(g2, dangerState, x, y);
        
        // Draw safety scores for different rows
        drawSafetyScores(g2, model, playerRow, x, y);
        
        // Draw performance metrics
        String metrics = fuzzyController.getPerformanceMetrics();
        g2.setColor(Color.WHITE);
        g2.setFont(new Font("Monospaced", Font.PLAIN, 11));
        g2.drawString(metrics, x, y + GRID_HEIGHT * CELL_SIZE + 20);
        
        // Restore original graphics state
        g2.setStroke(originalStroke);
        g2.setComposite(originalComposite);
        g2.setFont(originalFont);
    }
    
    /**
     * Draws the background grid for the visualization.
     */
    private void drawGridBackground(Graphics2D g2, int x, int y) {
        // Draw semi-transparent background
        g2.setColor(new Color(0, 0, 0, 180));
        g2.fillRect(x, y, GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE);
        
        // Draw grid lines
        g2.setColor(new Color(100, 100, 100, 180));
        for (int i = 0; i <= GRID_HEIGHT; i++) {
            g2.drawLine(x, y + i * CELL_SIZE, x + GRID_WIDTH * CELL_SIZE, y + i * CELL_SIZE);
        }
        
        for (int i = 0; i <= GRID_WIDTH; i++) {
            g2.drawLine(x + i * CELL_SIZE, y, x + i * CELL_SIZE, y + GRID_HEIGHT * CELL_SIZE);
        }
        
        // Label the grid
        g2.setFont(labelFont);
        g2.setColor(Color.WHITE);
        
        // Column labels (look ahead columns)
        for (int i = 0; i < LOOK_AHEAD; i++) {
            g2.drawString("+" + (i+1), x + (i+1) * CELL_SIZE + 5, y - 5);
        }
        
        // Row labels (every 5 rows)
        for (int i = 0; i < GRID_HEIGHT; i += 5) {
            g2.drawString(String.valueOf(i), x - 15, y + i * CELL_SIZE + 10);
        }
    }
    
    /**
     * Draws the tunnel centers detected by the fuzzy controller.
     */
    private void drawTunnelCenters(Graphics2D g2, int[] tunnelCenters, int x, int y) {
        g2.setColor(TUNNEL_COLOR);
        for (int i = 0; i < tunnelCenters.length; i++) {
            int centerY = tunnelCenters[i];
            if (centerY >= 0 && centerY < MODEL_HEIGHT) {
                g2.fillOval(
                    x + (i+1) * CELL_SIZE + CELL_SIZE/4, 
                    y + centerY * CELL_SIZE + CELL_SIZE/4, 
                    CELL_SIZE/2, 
                    CELL_SIZE/2
                );
            }
        }
        
        // Draw tunnel center label
        g2.setFont(labelFont);
        g2.drawString("Tunnel Centers", x + 5, y + GRID_HEIGHT * CELL_SIZE + 5);
    }
    
    /**
     * Draws the target row the controller is aiming for.
     */
    private void drawTargetRow(Graphics2D g2, int targetRow, int x, int y) {
        if (targetRow >= 0 && targetRow < MODEL_HEIGHT) {
            // Draw a line indicating the target row
            g2.setColor(TARGET_COLOR);
            g2.setStroke(new BasicStroke(2.0f));
            g2.drawLine(
                x, 
                y + targetRow * CELL_SIZE + CELL_SIZE/2, 
                x + GRID_WIDTH * CELL_SIZE, 
                y + targetRow * CELL_SIZE + CELL_SIZE/2
            );
            
            // Label the target row
            g2.setFont(labelFont);
            g2.drawString("Target: " + targetRow, x + GRID_WIDTH * CELL_SIZE + 5, y + targetRow * CELL_SIZE + 5);
        }
    }
    
    /**
     * Draws an indicator of the current danger state.
     */
    private void drawDangerIndicator(Graphics2D g2, boolean dangerState, int x, int y) {
        String status = dangerState ? "DANGER" : "SAFE";
        g2.setFont(new Font("SansSerif", Font.BOLD, 12));
        g2.setColor(dangerState ? Color.RED : Color.GREEN);
        g2.drawString(status, x + GRID_WIDTH * CELL_SIZE + 5, y + 15);
        
        // If in danger, add a red tint to the visualization
        if (dangerState) {
            g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.2f));
            g2.setColor(Color.RED);
            g2.fillRect(x, y, GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE);
            g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
        }
    }
    
    /**
     * Draws safety scores for different rows in the next columns.
     */
    private void drawSafetyScores(Graphics2D g2, LinkedList<byte[]> model, int playerRow, int x, int y) {
        // Draw player position
        g2.setColor(Color.CYAN);
        g2.fillRect(x, y + playerRow * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        
        // Calculate and visualize safety for the next column
        for (int row = 0; row < MODEL_HEIGHT; row++) {
            // Show the actual obstacles from the model
            int nextColumn = (PLAYER_COLUMN + 1) % model.size();
            if (model.get(nextColumn)[row] != 0) {
                g2.setColor(new Color(255, 0, 0, 200));
                g2.fillRect(
                    x + CELL_SIZE, 
                    y + row * CELL_SIZE, 
                    CELL_SIZE, 
                    CELL_SIZE
                );
            }
            
            // Color based on how safe/unsafe the row is
            boolean isSafeRow = row >= 2 && row <= MODEL_HEIGHT - 3;
            Color safetyColor = isSafeRow ? SAFE_COLOR : DANGER_COLOR;
            
            if (model.get(nextColumn)[row] == 0) {  // Only show safety for non-obstacle cells
                g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.3f));
                g2.setColor(safetyColor);
                g2.fillRect(
                    x + CELL_SIZE, 
                    y + row * CELL_SIZE, 
                    CELL_SIZE, 
                    CELL_SIZE
                );
                g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
            }
        }
        
        // Visualize the next columns (LOOK_AHEAD)
        for (int ahead = 2; ahead <= LOOK_AHEAD; ahead++) {
            int columnIndex = (PLAYER_COLUMN + ahead) % model.size();
            
            for (int row = 0; row < MODEL_HEIGHT; row++) {
                if (model.get(columnIndex)[row] != 0) {
                    g2.setColor(new Color(255, 0, 0, 150));  // Obstacles are red but fading with distance
                    g2.fillRect(
                        x + ahead * CELL_SIZE, 
                        y + row * CELL_SIZE, 
                        CELL_SIZE, 
                        CELL_SIZE
                    );
                }
            }
        }
    }
}