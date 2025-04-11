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
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;

import ie.atu.sw.game.AutopilotManager;
import ie.atu.sw.network.FlightData;
import ie.atu.sw.ui.PerformanceGraphPanel.MetricType;

/**
 * A comprehensive tool for analyzing flight performance data and visualizing
 * trends across multiple flights.
 */
public class FlightAnalyzer extends JFrame implements ActionListener {
    private static final long serialVersionUID = 1L;
    
    // Singleton instance
    private static FlightAnalyzer instance;
    
    // UI Components
    private DefaultTableModel flightHistoryModel;
    private PerformanceGraphPanel performanceGraphPanel;
    private JComboBox<MetricType> metricComboBox;
    private JLabel successRateLabel;
    private JLabel avgDurationLabel;
    private JLabel avgConfidenceLabel;
    private JLabel totalFlightsLabel;
    
    // Data
    private List<FlightData> flightHistory;
    
    // Visibility flag
    private boolean isVisible = false;
    
    /**
     * Gets the singleton instance of FlightAnalyzer.
     */
    public static synchronized FlightAnalyzer getInstance() {
        if (instance == null) {
            instance = new FlightAnalyzer();
        }
        return instance;
    }
    
    /**
     * Private constructor to enforce singleton pattern.
     */
    private FlightAnalyzer() {
        super("Flight Performance Analyzer");
        
        // Initialize UI
        initializeUI();
        
        // Handle window closing
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                isVisible = false;
                setVisible(false);
            }
        });
        
        // Set window properties
        setSize(900, 700);
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
        JPanel overviewPanel = createOverviewPanel();
        JPanel detailedDataPanel = createDetailedDataPanel();
        JPanel trendsPanel = createTrendsPanel();
        
        // Add panels to tabbed pane
        tabbedPane.addTab("Overview", overviewPanel);
        tabbedPane.addTab("Detailed Data", detailedDataPanel);
        tabbedPane.addTab("Performance Trends", trendsPanel);
        
        // Add tabbed pane to frame
        add(tabbedPane);
        
        // Control panel at bottom
        JPanel controlPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        
        JButton refreshButton = new JButton("Refresh Data");
        refreshButton.addActionListener(this);
        controlPanel.add(refreshButton);
        
        JButton closeButton = new JButton("Close");
        closeButton.addActionListener(e -> toggleVisibility());
        controlPanel.add(closeButton);
        
        add(controlPanel, BorderLayout.SOUTH);
    }
    
    /**
     * Creates the overview panel.
     */
    private JPanel createOverviewPanel() {
        JPanel panel = new JPanel(new BorderLayout(10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        
        // Summary statistics panel
        JPanel statsPanel = new JPanel(new GridLayout(2, 2, 10, 10));
        statsPanel.setBorder(BorderFactory.createTitledBorder("Summary Statistics"));
        
        // Create and style statistic panels
        successRateLabel = createStatsLabel("Success Rate: 0%");
        avgDurationLabel = createStatsLabel("Avg. Duration: 00:00");
        avgConfidenceLabel = createStatsLabel("Avg. Confidence: 0.0%");
        totalFlightsLabel = createStatsLabel("Total Flights: 0");
        
        statsPanel.add(successRateLabel);
        statsPanel.add(avgDurationLabel);
        statsPanel.add(avgConfidenceLabel);
        statsPanel.add(totalFlightsLabel);
        
        // Recent flights list
        String[] columns = {
            "Date", "Duration", "Actions", "Avg. Conf.", "Min. Conf.", "Result"
        };
        flightHistoryModel = new DefaultTableModel(columns, 0);
        
        JTable recentFlightsTable = new JTable(flightHistoryModel) {
            private static final long serialVersionUID = 1L;
            
            @Override
            public boolean isCellEditable(int row, int column) {
                return false; // Make table non-editable
            }
        };
        
        // Set custom renderers for better display
        DefaultTableCellRenderer centerRenderer = new DefaultTableCellRenderer();
        centerRenderer.setHorizontalAlignment(JLabel.CENTER);
        
        for (int i = 0; i < recentFlightsTable.getColumnCount(); i++) {
            recentFlightsTable.getColumnModel().getColumn(i).setCellRenderer(centerRenderer);
        }
        
        // Custom renderer for the "Result" column to highlight success/crash
        DefaultTableCellRenderer resultRenderer = new DefaultTableCellRenderer() {
            private static final long serialVersionUID = 1L;
            
            @Override
            public void setValue(Object value) {
                if (value != null) {
                    String text = value.toString();
                    setText(text);
                    
                    if (text.equals("Success")) {
                        setBackground(new Color(200, 255, 200)); // Light green
                    } else if (text.equals("Crashed")) {
                        setBackground(new Color(255, 200, 200)); // Light red
                    } else {
                        setBackground(Color.WHITE);
                    }
                }
            }
        };
        resultRenderer.setHorizontalAlignment(JLabel.CENTER);
        recentFlightsTable.getColumnModel().getColumn(5).setCellRenderer(resultRenderer);
        
        JScrollPane scrollPane = new JScrollPane(recentFlightsTable);
        scrollPane.setBorder(BorderFactory.createTitledBorder("Recent Flights"));
        
        // Add components to panel
        panel.add(statsPanel, BorderLayout.NORTH);
        panel.add(scrollPane, BorderLayout.CENTER);
        
        return panel;
    }
    
    /**
     * Creates a styled label for statistics.
     */
    private JLabel createStatsLabel(String text) {
        JLabel label = new JLabel(text, JLabel.CENTER);
        label.setFont(new Font("SansSerif", Font.BOLD, 14));
        label.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(Color.LIGHT_GRAY),
                BorderFactory.createEmptyBorder(10, 10, 10, 10)));
        return label;
    }
    
    /**
     * Creates the detailed data panel.
     */
    private JPanel createDetailedDataPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        
        // Full flight history table
        String[] columns = {
            "Date", "Duration", "Actions", "Up %", "Stay %", 
            "Down %", "Avg. Conf.", "Min. Conf.", "Result"
        };
        DefaultTableModel detailedModel = new DefaultTableModel(columns, 0);
        
        JTable detailedTable = new JTable(detailedModel) {
            private static final long serialVersionUID = 1L;
            
            @Override
            public boolean isCellEditable(int row, int column) {
                return false; // Make table non-editable
            }
        };
        
        // Set custom renderers
        DefaultTableCellRenderer centerRenderer = new DefaultTableCellRenderer();
        centerRenderer.setHorizontalAlignment(JLabel.CENTER);
        
        for (int i = 0; i < detailedTable.getColumnCount(); i++) {
            detailedTable.getColumnModel().getColumn(i).setCellRenderer(centerRenderer);
        }
        
        // Custom renderer for the "Result" column
        DefaultTableCellRenderer resultRenderer = new DefaultTableCellRenderer() {
            private static final long serialVersionUID = 1L;
            
            @Override
            public void setValue(Object value) {
                if (value != null) {
                    String text = value.toString();
                    setText(text);
                    
                    if (text.equals("Success")) {
                        setBackground(new Color(200, 255, 200)); // Light green
                    } else if (text.equals("Crashed")) {
                        setBackground(new Color(255, 200, 200)); // Light red
                    } else {
                        setBackground(Color.WHITE);
                    }
                }
            }
        };
        resultRenderer.setHorizontalAlignment(JLabel.CENTER);
        detailedTable.getColumnModel().getColumn(8).setCellRenderer(resultRenderer);
        
        JScrollPane scrollPane = new JScrollPane(detailedTable);
        
        // Filter options
        JPanel filterPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        filterPanel.setBorder(BorderFactory.createTitledBorder("Filters"));
        
        JRadioButton allFlightsRadio = new JRadioButton("All Flights");
        JRadioButton successFlightsRadio = new JRadioButton("Successful Flights");
        JRadioButton crashedFlightsRadio = new JRadioButton("Crashed Flights");
        
        ButtonGroup filterGroup = new ButtonGroup();
        filterGroup.add(allFlightsRadio);
        filterGroup.add(successFlightsRadio);
        filterGroup.add(crashedFlightsRadio);
        
        allFlightsRadio.setSelected(true);
        
        filterPanel.add(allFlightsRadio);
        filterPanel.add(successFlightsRadio);
        filterPanel.add(crashedFlightsRadio);
        
        // Export options
        JButton exportButton = new JButton("Export Data");
        filterPanel.add(exportButton);
        
        // Add action listeners
        allFlightsRadio.addActionListener(e -> filterDetailedTable(detailedModel, null));
        successFlightsRadio.addActionListener(e -> filterDetailedTable(detailedModel, false));
        crashedFlightsRadio.addActionListener(e -> filterDetailedTable(detailedModel, true));
        
        exportButton.addActionListener(e -> exportData());
        
        // Add components to panel
        panel.add(filterPanel, BorderLayout.NORTH);
        panel.add(scrollPane, BorderLayout.CENTER);
        
        return panel;
    }
    
    /**
     * Creates the trends panel.
     */
    private JPanel createTrendsPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        
        // Performance graph
        performanceGraphPanel = new PerformanceGraphPanel();
        performanceGraphPanel.setPreferredSize(new Dimension(800, 400));
        
        // Metric selection panel
        JPanel metricPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        metricPanel.setBorder(BorderFactory.createTitledBorder("Metric"));
        
        JLabel metricLabel = new JLabel("Select Metric: ");
        metricComboBox = new JComboBox<>(MetricType.values());
        metricComboBox.setSelectedItem(MetricType.AVG_CONFIDENCE);
        
        metricComboBox.addActionListener(e -> 
            performanceGraphPanel.setMetricType((MetricType) metricComboBox.getSelectedItem()));
        
        metricPanel.add(metricLabel);
        metricPanel.add(metricComboBox);
        
        // Add components to panel
        panel.add(metricPanel, BorderLayout.NORTH);
        panel.add(performanceGraphPanel, BorderLayout.CENTER);
        
        return panel;
    }
    
    /**
     * Filters the detailed table by crash status.
     * 
     * @param model The table model to filter
     * @param crashed null for all flights, true for crashed, false for successful
     */
    private void filterDetailedTable(DefaultTableModel model, Boolean crashed) {
        if (flightHistory == null) return;
        
        // Clear table
        model.setRowCount(0);
        
        // Add filtered flights
        for (int i = flightHistory.size() - 1; i >= 0; i--) {
            FlightData flight = flightHistory.get(i);
            
            // Apply filter
            if (crashed != null && flight.isCrashed() != crashed) {
                continue;
            }
            
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
            model.addRow(new Object[] {
                timestamp, duration, actions, upPercent, stayPercent, 
                downPercent, avgConf, minConf, result
            });
        }
    }
    
    /**
     * Exports the flight data to a CSV file.
     */
    private void exportData() {
        JOptionPane.showMessageDialog(this, 
                "This feature would export flight data to a CSV file.",
                "Export Data", 
                JOptionPane.INFORMATION_MESSAGE);
    }
    
    /**
     * Refreshes all data components with the latest flight history.
     */
    public void refreshData() {
        // Get latest flight history
        flightHistory = AutopilotManager.getInstance().getFlightHistory();
        
        if (flightHistory == null || flightHistory.isEmpty()) {
            // Clear tables and graphs
            flightHistoryModel.setRowCount(0);
            performanceGraphPanel.updateData(null);
            
            // Reset statistics
            successRateLabel.setText("Success Rate: 0%");
            avgDurationLabel.setText("Avg. Duration: 00:00");
            avgConfidenceLabel.setText("Avg. Confidence: 0.0%");
            totalFlightsLabel.setText("Total Flights: 0");
            
            return;
        }
        
        // Update overview table (recent flights)
        flightHistoryModel.setRowCount(0);
        
        // Show only the most recent flights (up to 10)
        int startIndex = Math.max(0, flightHistory.size() - 10);
        for (int i = flightHistory.size() - 1; i >= startIndex; i--) {
            FlightData flight = flightHistory.get(i);
            
            // Format the data
            String timestamp = flight.getTimestamp() != null ? 
                    flight.getTimestamp().toString() : "In Progress";
            String duration = FlightData.formatTime(flight.getFlightDuration());
            String actions = String.valueOf(flight.getTotalActions());
            String avgConf = String.format("%.1f%%", flight.getAvgConfidence() * 100);
            String minConf = String.format("%.1f%%", flight.getMinConfidence() * 100);
            String result = flight.isCrashed() ? "Crashed" : "Success";
            
            // Add row to table
            flightHistoryModel.addRow(new Object[] {
                timestamp, duration, actions, avgConf, minConf, result
            });
        }
        
        // Update detailed table (all tabs share the same model)
        JTabbedPane tabbedPane = (JTabbedPane) getContentPane().getComponent(0);
        JPanel detailedPanel = (JPanel) tabbedPane.getComponentAt(1);
        JScrollPane scrollPane = (JScrollPane) detailedPanel.getComponent(1);
        JTable detailedTable = (JTable) scrollPane.getViewport().getView();
        DefaultTableModel detailedModel = (DefaultTableModel) detailedTable.getModel();
        
        filterDetailedTable(detailedModel, null);
        
        // Update performance graph
        performanceGraphPanel.updateData(flightHistory);
        performanceGraphPanel.setMetricType((MetricType) metricComboBox.getSelectedItem());
        
        // Calculate statistics
        int totalFlights = flightHistory.size();
        int successfulFlights = 0;
        long totalDuration = 0;
        double totalConfidence = 0;
        
        for (FlightData flight : flightHistory) {
            if (!flight.isCrashed()) {
                successfulFlights++;
            }
            
            totalDuration += flight.getFlightDuration();
            totalConfidence += flight.getAvgConfidence();
        }
        
        double successRate = totalFlights > 0 ? 
                (double) successfulFlights / totalFlights * 100 : 0;
        
        double avgDuration = totalFlights > 0 ? 
                (double) totalDuration / totalFlights : 0;
        
        double avgConfidence = totalFlights > 0 ? 
                totalConfidence / totalFlights * 100 : 0;
        
        // Update statistics labels
        successRateLabel.setText(String.format("Success Rate: %.1f%%", successRate));
        avgDurationLabel.setText("Avg. Duration: " + FlightData.formatTime((long) avgDuration));
        avgConfidenceLabel.setText(String.format("Avg. Confidence: %.1f%%", avgConfidence));
        totalFlightsLabel.setText("Total Flights: " + totalFlights);
        
        // Set success rate label color based on rate
        if (successRate >= 80) {
            successRateLabel.setForeground(new Color(0, 150, 0)); // Dark green
        } else if (successRate >= 50) {
            successRateLabel.setForeground(new Color(200, 150, 0)); // Gold
        } else {
            successRateLabel.setForeground(new Color(200, 0, 0)); // Dark red
        }
    }
    
    /**
     * ActionListener for the refresh button.
     */
    @Override
    public void actionPerformed(ActionEvent e) {
        refreshData();
    }
    
    /**
     * Toggles the visibility of the window.
     */
    public void toggleVisibility() {
        isVisible = !isVisible;
        setVisible(isVisible);
        
        if (isVisible) {
            refreshData();
        }
    }
}