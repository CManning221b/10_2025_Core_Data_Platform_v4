// backend/methods_application/javascript_methods/sensor_bounds_check.js

function checkSensorBounds(value, minThreshold, maxThreshold) {
    const numValue = parseFloat(value);
    const numMin = parseFloat(minThreshold);
    const numMax = parseFloat(maxThreshold);

    if (numValue < numMin) {
        return {
            valid: false,
            status: "below_range",
            value: numValue,
            violation_amount: Math.round((numMin - numValue) * 100) / 100,
            threshold_violated: "minimum"
        };
    } else if (numValue > numMax) {
        return {
            valid: false,
            status: "above_range",
            value: numValue,
            violation_amount: Math.round((numValue - numMax) * 100) / 100,
            threshold_violated: "maximum"
        };
    } else {
        return {
            valid: true,
            status: "within_range",
            value: numValue,
            margin: Math.round(Math.min(numValue - numMin, numMax - numValue) * 100) / 100
        };
    }
}

function checkSequenceBounds(values, minThreshold, maxThreshold) {
    const results = values.map(value => checkSensorBounds(value, minThreshold, maxThreshold));

    const totalMeasurements = results.length;
    const validMeasurements = results.filter(r => r.valid).length;
    const violations = results.filter(r => !r.valid);
    const violationCount = violations.length;
    const violationPercentage = Math.round((violationCount / totalMeasurements * 100) * 10) / 10;

    let sequenceStatus;
    if (violationPercentage === 0) {
        sequenceStatus = "all_within_range";
    } else if (violationPercentage < 10) {
        sequenceStatus = "mostly_within_range";
    } else if (violationPercentage < 50) {
        sequenceStatus = "some_violations";
    } else {
        sequenceStatus = "many_violations";
    }

    return {
        measurement_type: 'sequence',
        is_valid: violationCount === 0,
        sequence_status: sequenceStatus,
        total_measurements: totalMeasurements,
        valid_measurements: validMeasurements,
        violation_count: violationCount,
        violation_percentage: violationPercentage,
        violations: violations.slice(0, 5), // Include up to 5 example violations
        analysis_summary: `Sequence: ${violationCount}/${totalMeasurements} violations (${violationPercentage}%)`
    };
}

// Get command line arguments
const args = process.argv.slice(2);

if (args.length === 3) {
    // Single value mode (original functionality)
    const value = args[0];
    const minThreshold = args[1];
    const maxThreshold = args[2];

    const result = checkSensorBounds(value, minThreshold, maxThreshold);
    console.log(JSON.stringify(result));

} else if (args.length === 1 && args[0].startsWith('{')) {
    // Sequence mode - expects JSON input with {values: [...], min: X, max: Y}
    try {
        const input = JSON.parse(args[0]);
        const result = checkSequenceBounds(input.values, input.min, input.max);
        console.log(JSON.stringify(result));
    } catch (error) {
        console.error("Error parsing JSON input:", error.message);
        process.exit(1);
    }

} else {
    console.error("Usage:");
    console.error("  Single value: node sensor_bounds_check.js <value> <min> <max>");
    console.error("  Sequence: node sensor_bounds_check.js '{\"values\":[1,2,3],\"min\":0,\"max\":5}'");
    process.exit(1);
}