package ie.atu.sw.ui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Stroke;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JPanel;

/**
 * A panel for visualizing the confidence history over time.
 */
public class ConfidenceHistoryPanel extends JPanel {
    private static final long serialVersionUID = 1L;
    
    // Data
    private List<Double> confidenceHistory = new ArrayList<>();
    
    // Visualization constants
    private static final int PADDING = 30;
    private static final int AXIS_PADDING = 10;
    private static final int MAX_POINTS = 100;
    private static final Color GRAPH_COLOR = new Color(0, 120, 200);
    private static final Color DANGER_COLOR = new Color(255, 80, 80);
    private static final Color AVERAGE_COLOR = new Color(0, 180, 0);
    private static final Stroke LINE_STROKE = new BasicStroke(2f);
    private static final Stroke DASHED_STROKE = new BasicStroke(1.0f, 
            BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 
            10.0f, new float[]{5.0f}, 0.0f);
    
    // Thresholds
    private static final double DANGER_THRESHOLD = 0.6; // 60% confidence
    
    /**
     * Creates a new confidence history panel.
     */
    public ConfidenceHistoryPanel() {
        setBackground(Color.WHITE);
    }
    
    /**
     * Updates the confidence history data.
     * 
     * @param history The new confidence history
     */
    public void updateData(List<Double> history) {
        if (history != null) {
            // Make a copy to avoid concurrent modification
            confidenceHistory = new ArrayList<>(history);
            
            // Limit the number of points to avoid overwhelming the chart
            if (confidenceHistory.size() > MAX_POINTS) {
                confidenceHistory = confidenceHistory.subList(
                        confidenceHistory.size() - MAX_POINTS, 
                        confidenceHistory.size());
            }
            
            // Trigger repaint
            repaint();
        }
    }
    
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        
        // Enable anti-aliasing for smoother rendering
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        
        // Get panel dimensions
        int width = getWidth();
        int height = getHeight();
        
        // Draw background
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, width, height);
        
        // Draw title
        g2.setColor(Color.BLACK);
        g2.setFont(new Font("SansSerif", Font.BOLD, 14));
        String title = "Neural Network Confidence History";
        int titleWidth = g2.getFontMetrics().stringWidth(title);
        g2.drawString(title, (width - titleWidth) / 2, PADDING / 2);
        
        // Draw axes
        g2.setColor(Color.BLACK);
        
        // X axis
        g2.drawLine(PADDING, height - PADDING, width - PADDING, height - PADDING);
        
        // Y axis
        g2.drawLine(PADDING, PADDING, PADDING, height - PADDING);
        
        // Calculate available plotting area
        int plotWidth = width - 2 * PADDING;
        int plotHeight = height - 2 * PADDING;
        
        // Draw danger threshold line
        int dangerY = height - PADDING - (int)(DANGER_THRESHOLD * plotHeight);
        g2.setColor(DANGER_COLOR);
        g2.setStroke(DASHED_STROKE);
        g2.drawLine(PADDING, dangerY, width - PADDING, dangerY);
        g2.drawString("Danger Threshold (60%)", PADDING + AXIS_PADDING, dangerY - 5);
        
        // Draw axis labels
        g2.setColor(Color.BLACK);
        g2.setFont(new Font("SansSerif", Font.PLAIN, 12));
        
        // Y-axis labels (confidence percentages)
        for (int i = 0; i <= 10; i++) {
            int y = height - PADDING - (i * plotHeight / 10);
            g2.drawLine(PADDING - 5, y, PADDING, y);
            g2.drawString(i * 10 + "%", PADDING - 30, y + 5);
        }
        
        // If no data, show message and return
        if (confidenceHistory.isEmpty()) {
            g2.setColor(Color.GRAY);
            String message = "No confidence data available";
            int messageWidth = g2.getFontMetrics().stringWidth(message);
            g2.drawString(message, (width - messageWidth) / 2, height / 2);
            return;
        }
        
        // Draw data points
        g2.setColor(GRAPH_COLOR);
        g2.setStroke(LINE_STROKE);
        
        int totalPoints = confidenceHistory.size();
        double pointSpacing = (double) plotWidth / (totalPoints > 1 ? totalPoints - 1 : 1);
        
        // Draw confidence line
        for (int i = 0; i < totalPoints - 1; i++) {
            int x1 = PADDING + (int)(i * pointSpacing);
            int y1 = height - PADDING - (int)(confidenceHistory.get(i) * plotHeight);
            int x2 = PADDING + (int)((i + 1) * pointSpacing);
            int y2 = height - PADDING - (int)(confidenceHistory.get(i + 1) * plotHeight);
            
            // Draw line segment
            g2.drawLine(x1, y1, x2, y2);
            
            // Draw point at low confidence values with alert color
            if (confidenceHistory.get(i) < DANGER_THRESHOLD) {
                g2.setColor(DANGER_COLOR);
                g2.fillOval(x1 - 3, y1 - 3, 6, 6);
                g2.setColor(GRAPH_COLOR);
            } else {
                g2.fillOval(x1 - 2, y1 - 2, 4, 4);
            }
        }
        
        // Draw the last point
        int lastX = PADDING + (int)((totalPoints - 1) * pointSpacing);
        int lastY = height - PADDING - (int)(confidenceHistory.get(totalPoints - 1) * plotHeight);
        
        if (confidenceHistory.get(totalPoints - 1) < DANGER_THRESHOLD) {
            g2.setColor(DANGER_COLOR);
            g2.fillOval(lastX - 3, lastY - 3, 6, 6);
        } else {
            g2.fillOval(lastX - 2, lastY - 2, 4, 4);
        }
        
        // Calculate and draw average line
        double avgConfidence = confidenceHistory.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        int avgY = height - PADDING - (int)(avgConfidence * plotHeight);
        
        g2.setColor(AVERAGE_COLOR);
        g2.setStroke(DASHED_STROKE);
        g2.drawLine(PADDING, avgY, width - PADDING, avgY);
        g2.drawString(String.format("Average: %.1f%%", avgConfidence * 100), 
                width - PADDING - 120, avgY - 5);
        
        // Draw legend
        int legendX = PADDING + 20;
        int legendY = PADDING + 20;
        
        g2.setColor(GRAPH_COLOR);
        g2.setStroke(LINE_STROKE);
        g2.drawLine(legendX, legendY, legendX + 30, legendY);
        g2.fillOval(legendX + 15 - 2, legendY - 2, 4, 4);
        g2.setColor(Color.BLACK);
        g2.drawString("Confidence", legendX + 40, legendY + 5);
        
        legendY += 20;
        g2.setColor(DANGER_COLOR);
        g2.setStroke(DASHED_STROKE);
        g2.drawLine(legendX, legendY, legendX + 30, legendY);
        g2.setColor(Color.BLACK);
        g2.drawString("Danger Threshold", legendX + 40, legendY + 5);
        
        legendY += 20;
        g2.setColor(AVERAGE_COLOR);
        g2.setStroke(DASHED_STROKE);
        g2.drawLine(legendX, legendY, legendX + 30, legendY);
        g2.setColor(Color.BLACK);
        g2.drawString("Average", legendX + 40, legendY + 5);
    }
    
    @Override
    public Dimension getPreferredSize() {
        return new Dimension(600, 300);
    }
}