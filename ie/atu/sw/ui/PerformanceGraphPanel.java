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

import ie.atu.sw.network.FlightData;

/**
 * A panel for visualizing performance metrics across multiple flights.
 */
public class PerformanceGraphPanel extends JPanel {
    private static final long serialVersionUID = 1L;
    
    // Types of metrics to display
    public enum MetricType {
        FLIGHT_DURATION("Flight Duration (s)"),
        AVG_CONFIDENCE("Average Confidence (%)"),
        MIN_CONFIDENCE("Minimum Confidence (%)"),
        UP_PERCENTAGE("Up Actions (%)"),
        STAY_PERCENTAGE("Stay Actions (%)"),
        DOWN_PERCENTAGE("Down Actions (%)");
        
        private final String label;
        
        MetricType(String label) {
            this.label = label;
        }
        
        public String getLabel() {
            return label;
        }
    }
    
    // Data
    private List<FlightData> flightHistory = new ArrayList<>();
    private MetricType metricType = MetricType.AVG_CONFIDENCE;
    
    // Visualization constants
    private static final int PADDING = 40;
    private static final int AXIS_PADDING = 10;
    private static final Color GRAPH_COLOR = new Color(0, 120, 200);
    private static final Color SUCCESS_COLOR = new Color(0, 180, 0);
    private static final Color CRASH_COLOR = new Color(200, 0, 0);
    private static final Color TREND_COLOR = new Color(255, 140, 0);
    private static final Stroke LINE_STROKE = new BasicStroke(2f);
    private static final Stroke DASHED_STROKE = new BasicStroke(1.0f, 
            BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 
            10.0f, new float[]{5.0f}, 0.0f);
    
    /**
     * Creates a new performance graph panel.
     */
    public PerformanceGraphPanel() {
        setBackground(Color.WHITE);
        setPreferredSize(new Dimension(600, 300));
    }
    
    /**
     * Updates the flight history data.
     * 
     * @param flightHistory The flight history data
     */
    public void updateData(List<FlightData> flightHistory) {
        if (flightHistory != null) {
            this.flightHistory = new ArrayList<>(flightHistory);
            repaint();
        }
    }
    
    /**
     * Sets the metric type to display.
     * 
     * @param type The metric type
     */
    public void setMetricType(MetricType type) {
        this.metricType = type;
        repaint();
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
        String title = "Flight Performance: " + metricType.getLabel();
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
        
        // If no data, show message and return
        if (flightHistory.isEmpty()) {
            g2.setColor(Color.GRAY);
            String message = "No flight history data available";
            int messageWidth = g2.getFontMetrics().stringWidth(message);
            g2.drawString(message, (width - messageWidth) / 2, height / 2);
            return;
        }
        
        // Find max value of selected metric for scaling
        double maxValue = getMaxValue();
        
        // Draw Y-axis labels
        g2.setColor(Color.BLACK);
        g2.setFont(new Font("SansSerif", Font.PLAIN, 12));
        
        int divisions = 5;
        for (int i = 0; i <= divisions; i++) {
            int y = height - PADDING - (i * plotHeight / divisions);
            g2.drawLine(PADDING - 5, y, PADDING, y);
            
            String label;
            if (metricType == MetricType.FLIGHT_DURATION) {
                label = String.valueOf((int)(maxValue * i / divisions));
            } else {
                label = String.format("%.1f%%", (maxValue * i / divisions));
            }
            
            g2.drawString(label, PADDING - 35, y + 5);
        }
        
        // Draw X-axis labels
        for (int i = 0; i < flightHistory.size(); i++) {
            int x = PADDING + (i + 1) * plotWidth / flightHistory.size();
            g2.drawLine(x, height - PADDING, x, height - PADDING + 5);
            g2.drawString(String.valueOf(i + 1), x - 5, height - PADDING + 20);
        }
        
        // Draw data points
        double pointSpacing = (double) plotWidth / flightHistory.size();
        
        // Draw points and connect them
        for (int i = 0; i < flightHistory.size(); i++) {
            double value = getMetricValue(flightHistory.get(i));
            
            int x = PADDING + (int)((i + 0.5) * pointSpacing);
            int y = height - PADDING - (int)((value / maxValue) * plotHeight);
            
            // Draw point
            if (flightHistory.get(i).isCrashed()) {
                g2.setColor(CRASH_COLOR);
            } else {
                g2.setColor(SUCCESS_COLOR);
            }
            
            g2.fillOval(x - 5, y - 5, 10, 10);
            
            // Connect points with lines
            if (i > 0) {
                double prevValue = getMetricValue(flightHistory.get(i - 1));
                int prevX = PADDING + (int)((i - 0.5) * pointSpacing);
                int prevY = height - PADDING - (int)((prevValue / maxValue) * plotHeight);
                
                g2.setColor(GRAPH_COLOR);
                g2.setStroke(LINE_STROKE);
                g2.drawLine(prevX, prevY, x, y);
            }
        }
        
        // Draw trend line (linear regression)
        if (flightHistory.size() >= 2) {
            drawTrendLine(g2, plotWidth, plotHeight, height, maxValue);
        }
        
        // Draw legend
        int legendX = PADDING + 20;
        int legendY = PADDING + 20;
        
        g2.setColor(GRAPH_COLOR);
        g2.setStroke(LINE_STROKE);
        g2.drawLine(legendX, legendY, legendX + 30, legendY);
        g2.setColor(Color.BLACK);
        g2.drawString("Metric Values", legendX + 40, legendY + 5);
        
        legendY += 20;
        g2.setColor(SUCCESS_COLOR);
        g2.fillOval(legendX + 15 - 4, legendY - 4, 8, 8);
        g2.setColor(Color.BLACK);
        g2.drawString("Successful Flight", legendX + 40, legendY + 5);
        
        legendY += 20;
        g2.setColor(CRASH_COLOR);
        g2.fillOval(legendX + 15 - 4, legendY - 4, 8, 8);
        g2.setColor(Color.BLACK);
        g2.drawString("Crashed Flight", legendX + 40, legendY + 5);
        
        legendY += 20;
        g2.setColor(TREND_COLOR);
        g2.setStroke(DASHED_STROKE);
        g2.drawLine(legendX, legendY, legendX + 30, legendY);
        g2.setColor(Color.BLACK);
        g2.drawString("Performance Trend", legendX + 40, legendY + 5);
    }
    
    /**
     * Calculates and draws a trend line using linear regression.
     */
    private void drawTrendLine(Graphics2D g2, int plotWidth, int plotHeight, int height, double maxValue) {
        // Prepare data for linear regression
        double[] x = new double[flightHistory.size()];
        double[] y = new double[flightHistory.size()];
        
        for (int i = 0; i < flightHistory.size(); i++) {
            x[i] = i;
            y[i] = getMetricValue(flightHistory.get(i));
        }
        
        // Calculate linear regression
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        int n = flightHistory.size();
        
        for (int i = 0; i < n; i++) {
            sumX += x[i];
            sumY += y[i];
            sumXY += x[i] * y[i];
            sumX2 += x[i] * x[i];
        }
        
        double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        double intercept = (sumY - slope * sumX) / n;
        
        // Draw trend line
        g2.setColor(TREND_COLOR);
        g2.setStroke(DASHED_STROKE);
        
        // Calculate start and end points
        double startY = intercept;
        double endY = slope * (n - 1) + intercept;
        
        int startX = PADDING + (int)(0.5 * plotWidth / n);
        int startYPixel = height - PADDING - (int)((startY / maxValue) * plotHeight);
        
        int endX = PADDING + (int)((n - 0.5) * plotWidth / n);
        int endYPixel = height - PADDING - (int)((endY / maxValue) * plotHeight);
        
        g2.drawLine(startX, startYPixel, endX, endYPixel);
    }
    
    /**
     * Gets the maximum value of the selected metric across all flights.
     */
    private double getMaxValue() {
        double max = 0;
        for (FlightData flight : flightHistory) {
            double value = getMetricValue(flight);
            if (value > max) {
                max = value;
            }
        }
        
        // Add a 10% margin to max for better visualization
        return max * 1.1;
    }
    
    /**
     * Gets the metric value for a specific flight.
     */
    private double getMetricValue(FlightData flight) {
        switch (metricType) {
            case FLIGHT_DURATION:
                return flight.getFlightDuration();
            case AVG_CONFIDENCE:
                return flight.getAvgConfidence() * 100;
            case MIN_CONFIDENCE:
                return flight.getMinConfidence() * 100;
            case UP_PERCENTAGE:
                return flight.getUpPercentage() * 100;
            case STAY_PERCENTAGE:
                return flight.getStayPercentage() * 100;
            case DOWN_PERCENTAGE:
                return flight.getDownPercentage() * 100;
            default:
                return 0;
        }
    }
}