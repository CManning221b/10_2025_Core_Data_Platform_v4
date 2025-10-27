// backend/methods_application/javascript_methods/sensor_characterization.js

function calculateTrend(dataPoints) {
    const n = dataPoints.length;

    if (n < 2) {
        return {
            slope: 0,
            intercept: 0,
            r_squared: 0,
            error: "Insufficient data points for trend calculation"
        };
    }

    // Extract x (time) and y (values) arrays
    const xValues = dataPoints.map(point => point.time);
    const yValues = dataPoints.map(point => point.value);

    // Calculate linear regression slope using least squares
    const sumX = xValues.reduce((sum, x) => sum + x, 0);
    const sumY = yValues.reduce((sum, y) => sum + y, 0);
    const sumXY = xValues.reduce((sum, x, i) => sum + x * yValues[i], 0);
    const sumX2 = xValues.reduce((sum, x) => sum + x * x, 0);

    // Calculate slope (m) and intercept (b) for y = mx + b
    const denominator = n * sumX2 - sumX * sumX;
    let slope = 0;
    if (denominator !== 0) {
        slope = (n * sumXY - sumX * sumY) / denominator;
    }

    const intercept = (sumY - slope * sumX) / n;

    // Calculate R-squared for trend strength
    const yMean = sumY / n;
    const ssTot = yValues.reduce((sum, y) => sum + Math.pow(y - yMean, 2), 0);
    const ssRes = yValues.reduce((sum, y, i) => {
        const predicted = slope * xValues[i] + intercept;
        return sum + Math.pow(y - predicted, 2);
    }, 0);

    const rSquared = ssTot !== 0 ? 1 - (ssRes / ssTot) : 0;

    return {
        slope: slope,
        intercept: intercept,
        r_squared: rSquared
    };
}

function calculateCorrelation(xValues, yValues) {
    if (xValues.length !== yValues.length || xValues.length < 2) {
        return 0.0;
    }

    const n = xValues.length;
    const sumX = xValues.reduce((sum, x) => sum + x, 0);
    const sumY = yValues.reduce((sum, y) => sum + y, 0);
    const sumXY = xValues.reduce((sum, x, i) => sum + x * yValues[i], 0);
    const sumX2 = xValues.reduce((sum, x) => sum + x * x, 0);
    const sumY2 = yValues.reduce((sum, y) => sum + y * y, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

    if (denominator === 0) {
        return 0.0;
    }

    return numerator / denominator;
}

function calculateTrendConsistency(trends) {
    if (trends.length < 2) {
        return 1.0;
    }

    // Calculate coefficient of variation (standard deviation / mean)
    const meanTrend = trends.reduce((sum, t) => sum + t, 0) / trends.length;

    if (meanTrend === 0) {
        return 1.0;
    }

    const variance = trends.reduce((sum, t) => sum + Math.pow(t - meanTrend, 2), 0) / trends.length;
    const stdDev = Math.sqrt(variance);

    // Convert to consistency score (1 = very consistent, 0 = very inconsistent)
    const cv = stdDev / Math.abs(meanTrend);
    const consistency = Math.max(0, 1 - cv);

    return consistency;
}

function characterizeSensorUsefulness(sensorId, sensorTrends, rulValues, correlation, dataCount) {
    // Calculate additional metrics
    const absCorrelation = Math.abs(correlation);
    const trendConsistency = calculateTrendConsistency(sensorTrends);
    const dataCoverage = dataCount;

    // Determine usefulness category based on correlation strength
    let usefulnessCategory, usefulnessScore;

    if (absCorrelation >= 0.4) {
        usefulnessCategory = "primary_indicator";
        usefulnessScore = absCorrelation;
    } else if (absCorrelation >= 0.2) {
        usefulnessCategory = "secondary_indicator";
        usefulnessScore = absCorrelation * 0.7; // Reduce score for secondary
    } else {
        usefulnessCategory = "noise_sensor";
        usefulnessScore = absCorrelation * 0.3; // Low score for noise
    }

    // Calculate early warning capability (higher correlation = better early warning)
    const earlyWarningCapability = Math.min(absCorrelation * 100, 95); // Cap at 95%

    // Determine correlation direction meaning
    let correlationMeaning;
    if (correlation > 0) {
        correlationMeaning = "increasing_trend_indicates_degradation";
    } else if (correlation < 0) {
        correlationMeaning = "decreasing_trend_indicates_degradation";
    } else {
        correlationMeaning = "no_clear_relationship";
    }

    return {
        sensor_id: sensorId,
        rul_correlation: Math.round(correlation * 10000) / 10000,
        correlation_strength: absCorrelation,
        usefulness_category: usefulnessCategory,
        usefulness_score: Math.round(usefulnessScore * 1000) / 1000,
        early_warning_capability: Math.round(earlyWarningCapability * 10) / 10,
        correlation_meaning: correlationMeaning,
        trend_consistency: Math.round(trendConsistency * 1000) / 1000,
        data_coverage: dataCoverage,
        engines_analyzed: sensorTrends.length,
        implementation: "javascript"
    };
}

// Enhanced processing for sensor characterization analysis
function processSensorCharacterization(analysisData) {
    try {
        const { sensorId, engineMeasurements, engineRulData } = analysisData;

        // Collect sensor trends and corresponding RUL values
        const sensorTrends = [];
        const rulValues = [];

        // Process each engine's measurement data
        for (const [engineId, measurementData] of Object.entries(engineMeasurements)) {
            if (!engineRulData[engineId]) {
                continue;
            }

            const dataPoints = measurementData.data_points;
            if (dataPoints.length < 2) {
                continue;
            }

            // Calculate trend for this engine's sensor data
            const trendResult = calculateTrend(dataPoints);

            // Use slope as the trend indicator
            sensorTrends.push(trendResult.slope);
            rulValues.push(engineRulData[engineId].rul_value);
        }

        if (sensorTrends.length < 3) {
            return {
                error: "Insufficient engine data for correlation analysis",
                minimum_required: 3,
                available: sensorTrends.length
            };
        }

        // Calculate correlation between sensor trends and RUL
        const correlation = calculateCorrelation(sensorTrends, rulValues);

        // Calculate characterization metrics
        const characterization = characterizeSensorUsefulness(
            sensorId,
            sensorTrends,
            rulValues,
            correlation,
            Object.keys(engineMeasurements).length
        );

        return characterization;

    } catch (error) {
        return {
            error: "Processing failed: " + error.message
        };
    }
}

// Get command line arguments
const args = process.argv.slice(2);

if (args.length === 1) {
    // Handle single argument - could be simple trend calculation (backwards compatibility)
    try {
        const dataPoints = JSON.parse(args[0]);

        // Check if it's the old format (array of data points) or new format (analysis object)
        if (Array.isArray(dataPoints)) {
            // Old format - simple trend calculation for backwards compatibility
            const result = calculateTrend(dataPoints);
            console.log(JSON.stringify(result));
        } else {
            // New format - sensor characterization analysis
            const result = processSensorCharacterization(dataPoints);
            console.log(JSON.stringify(result));
        }
    } catch (error) {
        console.error(JSON.stringify({
            error: "Error parsing input data: " + error.message
        }));
        process.exit(1);
    }
} else {
    console.error(JSON.stringify({
        error: "Usage: node sensor_characterization.js '<json_data>'"
    }));
    process.exit(1);
}