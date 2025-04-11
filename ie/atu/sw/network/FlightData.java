package ie.atu.sw.network;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;

/**
 * Unified flight data class that stores all metrics related to a flight.
 * Replaces FlightMetrics, FlightRecord, and parts of FlightRecorder.
 */
public class FlightData {
	// Basic flight metrics
	private final int totalActions;
	private final int upActions;
	private final int stayActions;
	private final int downActions;
	private final double avgConfidence;
	private final double minConfidence;

	// Complete flight info (may be null for in-progress flights)
	private final Date timestamp;
	private final long flightDuration;
	private final boolean crashed;

	// Real-time data (may be empty for completed flights)
	private final List<Double> confidenceHistory;

	/**
	 * Creates flight data for an ongoing flight.
	 */
	public FlightData(int totalActions, int upActions, int stayActions, int downActions, double avgConfidence,
			double minConfidence, List<Double> confidenceHistory) {
		this.totalActions = totalActions;
		this.upActions = upActions;
		this.stayActions = stayActions;
		this.downActions = downActions;
		this.avgConfidence = avgConfidence;
		this.minConfidence = minConfidence;
		this.confidenceHistory = confidenceHistory != null
				? Collections.unmodifiableList(new ArrayList<>(confidenceHistory))
				: Collections.emptyList();

		// These values are unknown for in-progress flights
		this.timestamp = null;
		this.flightDuration = 0;
		this.crashed = false;
	}

	/**
	 * Creates flight data for a completed flight.
	 */
	public FlightData(Date timestamp, long flightDuration, int totalActions, int upActions, int stayActions,
			int downActions, double avgConfidence, double minConfidence, boolean crashed) {
		this.timestamp = timestamp != null ? new Date(timestamp.getTime()) : new Date();
		this.flightDuration = flightDuration;
		this.totalActions = totalActions;
		this.upActions = upActions;
		this.stayActions = stayActions;
		this.downActions = downActions;
		this.avgConfidence = avgConfidence;
		this.minConfidence = minConfidence;
		this.crashed = crashed;
		this.confidenceHistory = Collections.emptyList(); // No history for completed flights
	}

	// Getters
	public int getTotalActions() {
		return totalActions;
	}

	public int getUpActions() {
		return upActions;
	}

	public int getStayActions() {
		return stayActions;
	}

	public int getDownActions() {
		return downActions;
	}

	public double getAvgConfidence() {
		return avgConfidence;
	}

	public double getMinConfidence() {
		return minConfidence;
	}

	public List<Double> getConfidenceHistory() {
		return confidenceHistory;
	}

	public boolean isCompleted() {
		return timestamp != null;
	}

	// Complete flight getters (may return null/0/false for in-progress flights)
	public Date getTimestamp() {
		return timestamp != null ? new Date(timestamp.getTime()) : null;
	}

	public long getFlightDuration() {
		return flightDuration;
	}

	public boolean isCrashed() {
		return crashed;
	}

	// Calculated metrics
	public double getUpPercentage() {
		return totalActions > 0 ? (double) upActions / totalActions : 0.0;
	}

	public double getStayPercentage() {
		return totalActions > 0 ? (double) stayActions / totalActions : 0.0;
	}

	public double getDownPercentage() {
		return totalActions > 0 ? (double) downActions / totalActions : 0.0;
	}

	/**
	 * Static utility method to format time in seconds to mm:ss format.
	 */
	public static String formatTime(long seconds) {
		return String.format("%02d:%02d", seconds / 60, seconds % 60);
	}
}