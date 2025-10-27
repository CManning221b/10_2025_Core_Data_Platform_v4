class ButtonInterface {
    constructor() {
        this.graphData = JSON.parse(document.getElementById('graph-data').textContent);
        this.outputContent = document.getElementById('cli-output');
        this.setupEventListeners();
        this.loadGraph();
    }

    setupEventListeners() {
        // Modal buttons
        document.getElementById('show-methods-modal')?.addEventListener('click', () => {
            this.showMethodsModal();
        });

        document.getElementById('show-warnings-modal')?.addEventListener('click', () => {
            this.showWarningsModal();
        });

        document.getElementById('show-insights-modal')?.addEventListener('click', () => {
            this.showInsightsModal();
        });

        document.getElementById('show-recommendations-modal')?.addEventListener('click', () => {
            this.showRecommendationsModal();
        });

        // Search functionality
        const nlSearch = document.getElementById('nl-search');
        const searchBtn = document.getElementById('search-btn');

        searchBtn?.addEventListener('click', () => {
            this.processSearch(nlSearch.value);
        });

        nlSearch?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.processSearch(e.target.value);
            }
        });

        // Graph controls
        document.getElementById('reset-view')?.addEventListener('click', () => {
            this.resetGraphView();
        });

        document.getElementById('fullscreen-graph')?.addEventListener('click', () => {
            this.toggleFullscreen();
        });

        // Modal close functionality
        this.setupModalClose();
    }

    setupModalClose() {
        const modal = document.getElementById('method-analysis-modal');
        const closeBtn = modal?.querySelector('.close');

        closeBtn?.addEventListener('click', () => {
            this.hideModal();
        });

        window.addEventListener('click', (event) => {
            if (event.target === modal) {
                this.hideModal();
            }
        });
    }

    processMarkdown(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // **bold** -> <strong>bold</strong>
            .replace(/\*(.*?)\*/g, '<em>$1</em>');             // *italic* -> <em>italic</em>
    }

    async processSearch(query) {
        if (!query.trim()) return;

        this.addOutput(`<h4>Search: "${query}"</h4>`);
        this.addOutput('<div class="loading">Processing search...</div>');

        try {
            const searchUrl = this.graphData.api_endpoints?.nl_search || '/text_interface/api/cli/search';

            const response = await fetch(searchUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    graph_id: this.graphData.graph_id
                })
            });

            if (!response.ok) {
                throw new Error(`Search API returned ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            if (result.status === 'success') {
                // Process markdown in the results
                const processedResults = this.processMarkdown(result.results);
                this.addOutput(`<div class="search-result">${processedResults}</div>`);

                if (result.visualization_path) {
                    this.loadSearchVisualization(result.visualization_path);
                }
            } else {
                this.addOutput(`<div class="error">Search failed: ${result.error || 'Unknown error'}</div>`);
            }
        } catch (error) {
            console.error('Search error:', error);
            this.addOutput(`<div class="error">Search error: ${error.message}</div>`);
        }

        document.getElementById('nl-search').value = '';
    }

    async showMethodsModal() {
        this.showModal('Method Executions', '<div class="loading">Loading method analysis...</div>');

        try {
            // Use the correct API endpoint
            const methodUrl = this.graphData.api_endpoints?.method_analysis || '/text_interface/api/method-analysis';

            const response = await fetch(methodUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ graph_id: this.graphData.graph_id })
            });

            if (!response.ok) {
                throw new Error(`Method analysis API returned ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.error) {
                this.updateModalBody(`<div class="error">Error: ${data.error}</div>`);
            } else {
                this.updateModalBody(this.formatMethodsContent(data));
            }
        } catch (error) {
            console.error('Method analysis error:', error);
            this.updateModalBody(`<div class="error">Error loading method analysis: ${error.message}</div>`);
        }
    }

// Enhanced formatInsightsContent for better styling
formatInsightsContent(output) {
    const lines = output.split('\n');
    let html = '<div class="insights-content">';

    let currentSection = '';
    let findings = [];
    let recommendations = [];

    for (let line of lines) {
        line = line.trim();

        if (line.includes('KEY INSIGHTS')) {
            html += '<div class="insights-header"><h3>Key Insights Analysis</h3></div>';
        } else if (line.includes('FINDINGS:')) {
            html += '<div class="section-divider"><h4>Key Findings</h4></div>';
            currentSection = 'findings';
        } else if (line.includes('RECOMMENDATIONS')) {
            const count = line.match(/\((\d+)\)/);
            const recCount = count ? count[1] : 'Multiple';
            html += `<div class="section-divider">
                        <h4>Recommendations</h4>
                        <span class="recommendation-count">${recCount} suggestions</span>
                    </div>`;
            currentSection = 'recommendations';
        } else if (line.startsWith('•') || line.startsWith('-')) {
            const item = line.substring(1).trim();

            if (currentSection === 'findings') {
                findings.push(item);
            } else if (currentSection === 'recommendations') {
                // Parse recommendation format: "Method Name (Priority) - Description"
                const recMatch = item.match(/^(.+?)\s*\((.+?)\s*Priority\)/);
                if (recMatch) {
                    const methodName = recMatch[1].trim();
                    const priority = recMatch[2].trim();
                    const description = item.substring(recMatch[0].length).replace(/^[\s-]+/, '');

                    recommendations.push({
                        method: methodName,
                        priority: priority,
                        description: description
                    });
                } else {
                    recommendations.push({
                        method: item,
                        priority: 'Medium',
                        description: ''
                    });
                }
            }
        }
    }

    // Render findings
    if (findings.length > 0) {
        html += '<div class="findings-grid">';
        findings.forEach((finding, index) => {
            const findingClass = finding.toLowerCase().includes('failed') || finding.toLowerCase().includes('error') ? 'finding-error' :
                                finding.toLowerCase().includes('warning') || finding.toLowerCase().includes('issue') ? 'finding-warning' :
                                finding.toLowerCase().includes('passed') || finding.toLowerCase().includes('good') ? 'finding-success' : 'finding-info';

            html += `<div class="finding-card ${findingClass}">
                        <div class="finding-number">${index + 1}</div>
                        <div class="finding-text">${finding}</div>
                    </div>`;
        });
        html += '</div>';
    } else {
        html += `<div class="no-findings">
                    <div class="info-icon">i</div>
                    <h3>No Specific Findings</h3>
                    <p>Analysis complete. No specific issues or insights to report at this time.</p>
                </div>`;
    }

    // Render recommendations
    if (recommendations.length > 0) {
        html += '<div class="recommendations-grid">';
        recommendations.forEach((rec, index) => {
            const priorityClass = rec.priority.toLowerCase() === 'high' ? 'priority-high' :
                                rec.priority.toLowerCase() === 'medium' ? 'priority-medium' : 'priority-low';

            html += `<div class="recommendation-card ${priorityClass}">
                        <div class="recommendation-header">
                            <div class="recommendation-title">${rec.method}</div>
                            <span class="priority-badge ${priorityClass}">${rec.priority}</span>
                        </div>
                        ${rec.description ? `<div class="recommendation-description">${rec.description}</div>` : ''}
                    </div>`;
        });
        html += '</div>';
    }

    html += '</div>';
    return html;
}

// Enhanced showInsightsModal
async showInsightsModal() {
    this.showModal('Key Insights', '<div class="loading">Loading insights...</div>');

    try {
        const commandUrl = this.graphData.api_endpoints?.cli_command || '/text_interface/api/cli/command';

        const response = await fetch(commandUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                command: 'insights',
                graph_id: this.graphData.graph_id
            })
        });

        if (!response.ok) {
            throw new Error(`Insights API returned ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        this.updateModalBody(this.formatInsightsContent(result.output));
    } catch (error) {
        console.error('Insights error:', error);
        this.updateModalBody(`<div class="error">Error loading insights: ${error.message}</div>`);
    }
}


    // Enhanced formatRecommendationsContent for better styling
formatRecommendationsContent(output) {
    const lines = output.split('\n');
    let html = '<div class="recommendations-content">';

    let currentSection = '';
    let recommended = [];
    let applicable = [];

    for (let line of lines) {
        line = line.trim();

        if (line.includes('METHOD RECOMMENDATIONS')) {
            const applicableMatch = line.match(/Applicable Methods: (\d+)/);
            const applicableCount = applicableMatch ? applicableMatch[1] : 'Multiple';
            html += `<div class="recommendations-header">
                        <h3>Method Recommendations</h3>
                        <div class="applicable-count">${applicableCount} methods available</div>
                    </div>`;
        } else if (line.includes('RECOMMENDED TO RUN')) {
            const count = line.match(/\((\d+)\)/);
            const recCount = count ? count[1] : '0';
            html += `<div class="section-divider">
                        <h4>Recommended to Execute</h4>
                        <span class="section-count">${recCount} high-priority methods</span>
                    </div>`;
            currentSection = 'recommended';
        } else if (line.includes('ALL APPLICABLE METHODS:')) {
            html += `<div class="section-divider">
                        <h4>All Available Methods</h4>
                        <span class="section-count">Complete method library</span>
                    </div>`;
            currentSection = 'applicable';
        } else if (line.startsWith('•') || line.startsWith('-')) {
            const item = line.substring(1).trim();

            if (currentSection === 'recommended') {
                // Parse: "Method Name - Priority Priority"
                const parts = item.split(' - ');
                if (parts.length >= 2) {
                    const methodName = parts[0].trim();
                    const priority = parts[1].replace(/Priority$/, '').trim();

                    recommended.push({
                        method: methodName,
                        priority: priority,
                        type: 'recommended'
                    });
                }
            } else if (currentSection === 'applicable') {
                applicable.push({
                    method: item,
                    priority: 'Available',
                    type: 'applicable'
                });
            }
        } else if (line.includes('Applies to:') && currentSection === 'recommended') {
            // Add applies to info to last recommendation
            if (recommended.length > 0) {
                const appliesTo = line.replace('Applies to:', '').trim();
                recommended[recommended.length - 1].appliesTo = appliesTo;
            }
        } else if (line.includes('Description:') && currentSection === 'recommended') {
            // Add description to last recommendation
            if (recommended.length > 0) {
                const description = line.replace('Description:', '').trim();
                recommended[recommended.length - 1].description = description;
            }
        }
    }

    // Handle "all methods executed" case
    if (output.includes('All applicable methods have been executed')) {
        html += `<div class="all-complete">
                    <div class="complete-icon">✓</div>
                    <h3>All Methods Executed</h3>
                    <p>Excellent! All applicable validation and analysis methods have been run on your data.</p>
                </div>`;
    }

    // Render recommended methods
    if (recommended.length > 0) {
        html += '<div class="methods-priority-grid">';
        recommended.forEach((method, index) => {
            const priorityClass = method.priority.toLowerCase() === 'high' ? 'priority-high' :
                                method.priority.toLowerCase() === 'medium' ? 'priority-medium' : 'priority-low';

            html += `<div class="method-recommendation-card ${priorityClass}">
                        <div class="method-card-header">
                            <div class="method-name">${method.method}</div>
                            <span class="method-priority-badge ${priorityClass}">${method.priority}</span>
                        </div>
                        ${method.appliesTo ? `<div class="method-applies-to">
                            <strong>Applies to:</strong> ${method.appliesTo}
                        </div>` : ''}
                        ${method.description ? `<div class="method-description">${method.description}</div>` : ''}
                    </div>`;
        });
        html += '</div>';
    }

    // Render applicable methods in a compact grid
    if (applicable.length > 0) {
        html += '<div class="applicable-methods-section">';
        html += '<div class="applicable-methods-grid">';

        // Show first 6 methods, then make rest collapsible
        const showCount = Math.min(applicable.length, 6);

        applicable.slice(0, showCount).forEach((method, index) => {
            html += `<div class="applicable-method-item">
                        <div class="method-bullet">•</div>
                        <div class="method-text">${method.method}</div>
                    </div>`;
        });

        if (applicable.length > 6) {
            html += `<div class="show-more-methods" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'; this.textContent = this.textContent.includes('Show') ? 'Show fewer methods' : 'Show ${applicable.length - 6} more methods'">
                        Show ${applicable.length - 6} more methods
                    </div>
                    <div class="additional-methods" style="display: none;">`;

            applicable.slice(6).forEach((method, index) => {
                html += `<div class="applicable-method-item">
                            <div class="method-bullet">•</div>
                            <div class="method-text">${method.method}</div>
                        </div>`;
            });

            html += '</div>';
        }

        html += '</div></div>';
    }

    html += '</div>';
    return html;
}

// Enhanced showRecommendationsModal
async showRecommendationsModal() {
    this.showModal('Recommendations', '<div class="loading">Loading recommendations...</div>');

    try {
        const commandUrl = this.graphData.api_endpoints?.cli_command || '/text_interface/api/cli/command';

        const response = await fetch(commandUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                command: 'recommendations',
                graph_id: this.graphData.graph_id
            })
        });

        if (!response.ok) {
            throw new Error(`Recommendations API returned ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        this.updateModalBody(this.formatRecommendationsContent(result.output));
    } catch (error) {
        console.error('Recommendations error:', error);
        this.updateModalBody(`<div class="error">Error loading recommendations: ${error.message}</div>`);
    }
}

// Add this method to handle collapsible sections
toggleCollapse(elementId) {
    const element = document.getElementById(elementId);
    const toggleBtn = element.previousElementSibling.querySelector('.collapse-toggle');

    if (element.classList.contains('collapsed')) {
        element.classList.remove('collapsed');
        toggleBtn.textContent = '▼';
    } else {
        element.classList.add('collapsed');
        toggleBtn.textContent = '▶';
    }
}

// Updated formatMethodsContent with collapsible sections
formatMethodsContent(data) {
    const summary = data.method_execution_summary;
    const executions = data.method_executions || [];
    const insights = data.method_insights || {};

    let html = `
        <div class="analysis-summary">
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-number">${summary.total_methods_run}</div>
                    <div class="summary-label">Methods Executed</div>
                </div>
                <div class="summary-card">
                    <div class="summary-number">${summary.total_validations}</div>
                    <div class="summary-label">Validations Run</div>
                </div>
                <div class="summary-card ${summary.warnings_found > 0 ? 'warning' : ''}">
                    <div class="summary-number">${summary.warnings_found}</div>
                    <div class="summary-label">Warnings Found</div>
                </div>
            </div>
        </div>
        
        <div class="methods-list">
            <h3>Executed Methods</h3>
    `;

    executions.forEach((method, index) => {
        const connections = method.connections || {};
        const results = connections.produced_results || [];

        html += `
            <div class="method-card">
                <div class="method-header">
                    <h4>${method.method_type}</h4>
                    <span class="method-id">ID: ${method.node_id}</span>
                </div>
                <div class="method-content">
                    <p><strong>Value:</strong> ${method.value}</p>
                    
                    <div class="method-stats">
                        <span class="stat">Results produced: ${results.length}</span>
                    </div>
        `;

        // Show method execution details with collapse
        const rawData = method.data || {};
        const excludeKeys = ['type', 'value', 'hierarchy'];
        const dataEntries = Object.entries(rawData).filter(([key]) => !excludeKeys.includes(key));

        if (dataEntries.length > 0) {
            const showCount = Math.min(dataEntries.length, 3);
            html += `<div class="method-data">
                        <div class="section-header" onclick="window.buttonInterface.toggleCollapse('method-details-${method.node_id}')">
                            <strong>Method Details</strong>
                            <span class="collapse-toggle">${dataEntries.length > 3 ? '▶' : '▼'}</span>
                        </div>
                        <div id="method-details-${method.node_id}" class="collapsible-content ${dataEntries.length > 3 ? 'collapsed' : ''}">`;

            // Show first 3 always
            dataEntries.slice(0, showCount).forEach(([key, value]) => {
                if (value && value.toString().trim()) {
                    html += `<div class="data-item">${key}: ${value}</div>`;
                }
            });

            // Show remaining in collapsible section
            if (dataEntries.length > 3) {
                dataEntries.slice(3).forEach(([key, value]) => {
                    if (value && value.toString().trim()) {
                        html += `<div class="data-item">${key}: ${value}</div>`;
                    }
                });
            }

            html += `</div></div>`;
        }

        // Show results with collapse
        if (results.length > 0) {
            const showCount = Math.min(results.length, 2);
            html += `<div class="method-results">
                        <div class="section-header" onclick="window.buttonInterface.toggleCollapse('results-${method.node_id}')">
                            <strong>Results Produced (${results.length})</strong>
                            <span class="collapse-toggle">${results.length > 2 ? '▶' : '▼'}</span>
                        </div>
                        <div id="results-${method.node_id}" class="collapsible-content ${results.length > 2 ? 'collapsed' : ''}">`;

            results.forEach((result, resultIndex) => {
                const resultData = result.data || {};
                const statusInfo = this.getResultStatusInfo(resultData);

                // Only show first 2 by default, rest are collapsed
                const isHidden = resultIndex >= 2 && results.length > 2;

                html += `<div class="result-item ${statusInfo.cssClass}" ${isHidden ? 'style="display: none;"' : ''}>
                            <div class="result-header">
                                <span class="result-type">${result.type}</span>
                                <span class="result-id">ID: ${result.node_id}</span>
                                ${statusInfo.badge ? `<span class="status-badge ${statusInfo.badge.type}">${statusInfo.badge.text}</span>` : ''}
                            </div>
                            <div class="result-value">${result.value}</div>`;

                // Show result details with nested collapse
                const resultEntries = Object.entries(resultData).filter(([key]) =>
                    !['type', 'value', 'hierarchy'].includes(key) &&
                    resultData[key] !== null &&
                    resultData[key] !== undefined &&
                    resultData[key].toString().trim()
                );

                if (resultEntries.length > 0) {
                    const importantFields = ['is_valid', 'validation_reason', 'has_error', 'has_warning', 'status'];
                    const priorityEntries = resultEntries.filter(([key]) => importantFields.includes(key));
                    const otherEntries = resultEntries.filter(([key]) => !importantFields.includes(key));

                    html += `<div class="result-details">`;

                    // Always show priority fields
                    priorityEntries.forEach(([key, value]) => {
                        const itemClass = this.getFieldCssClass(key, value);
                        html += `<div class="detail-item ${itemClass}">${key}: ${value}</div>`;
                    });

                    // Show other fields with collapse if more than 3
                    if (otherEntries.length > 0) {
                        const showOtherCount = Math.min(otherEntries.length, 3);

                        otherEntries.slice(0, showOtherCount).forEach(([key, value]) => {
                            html += `<div class="detail-item">${key}: ${value}</div>`;
                        });

                        if (otherEntries.length > 3) {
                            html += `<div class="expand-toggle" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'; this.textContent = this.textContent.includes('Show') ? 'Hide additional properties' : 'Show ${otherEntries.length - 3} more properties'">
                                        Show ${otherEntries.length - 3} more properties
                                    </div>
                                    <div class="additional-details" style="display: none;">`;

                            otherEntries.slice(3).forEach(([key, value]) => {
                                html += `<div class="detail-item">${key}: ${value}</div>`;
                            });

                            html += `</div>`;
                        }
                    }

                    html += `</div>`;
                }

                html += `</div>`;
            });

            html += `</div></div>`;
        }

        html += `</div></div>`;
    });

    html += '</div>';

    // Enhanced validation summary
    if (insights.validation_summary && Object.keys(insights.validation_summary).length > 0) {
        html += '<div class="validation-summary enhanced"><h3>Validation Summary</h3><div class="validation-grid">';
        Object.entries(insights.validation_summary).forEach(([status, count]) => {
            const statusClass = status === 'PASSED' ? 'success' : status === 'FAILED' ? 'error' : status === 'WARNING' ? 'warning' : 'info';
            const icon = status === 'PASSED' ? '✓' : status === 'WARNING' ? '⚠' : status === 'FAILED' ? '✗' : 'ℹ';
            html += `<div class="validation-card ${statusClass}">
                        <div class="validation-icon">${icon}</div>
                        <div class="validation-info">
                            <div class="validation-count">${count}</div>
                            <div class="validation-label">${status}</div>
                        </div>
                    </div>`;
        });
        html += '</div></div>';
    }

    return html;
}

// Enhanced formatWarningsContent for better styling
formatWarningsContent(output) {
    // Parse the command output to create a better formatted display
    const lines = output.split('\n');
    let html = '<div class="warnings-content">';

    let currentSection = '';
    for (let line of lines) {
        line = line.trim();

        if (line.includes('WARNINGS & ISSUES')) {
            html += '<div class="warnings-header"><h3>Warnings & Issues Analysis</h3></div>';
        } else if (line.includes('Critical Issues:')) {
            const count = line.split(':')[1].trim();
            html += `<div class="critical-summary">
                        <div class="critical-badge ${count === '0' ? 'success' : 'error'}">
                            <span class="critical-count">${count}</span>
                            <span class="critical-label">Critical Issues</span>
                        </div>
                    </div>`;
        } else if (line.includes('FAILED VALIDATIONS:')) {
            html += '<div class="section-divider"><h4>Failed Validations</h4></div>';
            currentSection = 'failed';
        } else if (line.includes('WARNING VALIDATIONS:')) {
            html += '<div class="section-divider"><h4>Warning Validations</h4></div>';
            currentSection = 'warning';
        } else if (line.includes('HIGH SEVERITY ALERTS:')) {
            html += '<div class="section-divider"><h4>High Severity Alerts</h4></div>';
            currentSection = 'high';
        } else if (line.startsWith('•') || line.startsWith('-')) {
            const item = line.substring(1).trim();
            const cssClass = currentSection === 'failed' ? 'error' : currentSection === 'warning' ? 'warning' : 'high';
            html += `<div class="warning-item ${cssClass}">
                        <span class="warning-bullet">•</span>
                        <span class="warning-text">${item}</span>
                    </div>`;
        }
    }

    if (output.includes('No critical issues found')) {
        html = `<div class="no-issues">
                    <div class="success-icon">✓</div>
                    <h3>All Clear!</h3>
                    <p>No critical issues found. All validations are passing.</p>
                </div>`;
    }

    html += '</div>';
    return html;
}

// Enhanced showWarningsModal
async showWarningsModal() {
    this.showModal('Warnings & Issues', '<div class="loading">Loading warnings...</div>');

    try {
        const commandUrl = this.graphData.api_endpoints?.cli_command || '/text_interface/api/cli/command';

        const response = await fetch(commandUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                command: 'warnings',
                graph_id: this.graphData.graph_id
            })
        });

        if (!response.ok) {
            throw new Error(`Warnings API returned ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        this.updateModalBody(this.formatWarningsContent(result.output));
    } catch (error) {
        console.error('Warnings error:', error);
        this.updateModalBody(`<div class="error">Error loading warnings: ${error.message}</div>`);
    }
}
    // Helper method to determine result status and styling
    getResultStatusInfo(resultData) {
        // Check for status indicator fields
        if ('is_valid' in resultData) {
            if (resultData.is_valid === true) {
                return {
                    cssClass: 'result-passed',
                    badge: { type: 'success', text: 'PASSED' }
                };
            } else if (resultData.is_valid === false) {
                return {
                    cssClass: 'result-failed',
                    badge: { type: 'error', text: 'FAILED' }
                };
            }
        }

        if (resultData.has_error) {
            return {
                cssClass: 'result-failed',
                badge: { type: 'error', text: 'ERROR' }
            };
        }

        if (resultData.has_warning) {
            return {
                cssClass: 'result-warning',
                badge: { type: 'warning', text: 'WARNING' }
            };
        }

        if ('status' in resultData) {
            const status = String(resultData.status).toUpperCase();
            if (['PASSED', 'PASS', 'SUCCESS'].includes(status)) {
                return {
                    cssClass: 'result-passed',
                    badge: { type: 'success', text: 'PASSED' }
                };
            } else if (['FAILED', 'FAIL', 'ERROR'].includes(status)) {
                return {
                    cssClass: 'result-failed',
                    badge: { type: 'error', text: 'FAILED' }
                };
            } else if (['WARNING', 'WARN'].includes(status)) {
                return {
                    cssClass: 'result-warning',
                    badge: { type: 'warning', text: 'WARNING' }
                };
            }
        }

        // No status indicators - it's informational data
        return {
            cssClass: 'result-info',
            badge: null
        };
    }

    // Helper method for field-specific CSS classes
    getFieldCssClass(key, value) {
        if (key === 'is_valid') {
            return value ? 'field-success' : 'field-error';
        }
        if (key === 'has_error' && value) {
            return 'field-error';
        }
        if (key === 'has_warning' && value) {
            return 'field-warning';
        }
        if (key === 'validation_reason' && value) {
            const reason = String(value).toLowerCase();
            if (reason.includes('anachronism') || reason.includes('violation') || reason.includes('error')) {
                return 'field-error';
            }
        }
        return '';
    }


    showModal(title, content) {
        const modal = document.getElementById('method-analysis-modal');
        const modalTitle = document.getElementById('modal-title');
        const modalBody = document.getElementById('modal-body');

        // Check if elements exist
        if (!modal || !modalTitle || !modalBody) {
            console.error('Modal elements not found');
            this.addOutput(`<div class="error">Modal not available. ${title}: Loading...</div>`);
            return;
        }

        modalTitle.textContent = title;
        modalBody.innerHTML = content;
        modal.style.display = 'block';
        document.body.style.overflow = 'hidden';
    }

    updateModalBody(content) {
        const modalBody = document.getElementById('modal-body');
        if (modalBody) {
            modalBody.innerHTML = content;
        }
    }

    hideModal() {
        const modal = document.getElementById('method-analysis-modal');
        if (modal) {
            modal.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    }

    addOutput(content) {
        // Remove welcome message if it exists
        const welcome = this.outputContent.querySelector('.output-welcome');
        if (welcome) {
            welcome.remove();
        }

        const outputDiv = document.createElement('div');
        outputDiv.className = 'output-result';
        outputDiv.innerHTML = content;

        this.outputContent.appendChild(outputDiv);
        this.outputContent.scrollTop = this.outputContent.scrollHeight;
    }

    async loadGraph() {
        const container = document.getElementById('graph-container');
        const loading = container.querySelector('.graph-loading');

        try {
            const graphId = this.graphData.graph_id;

            // Try to find existing visualization
            let pyvisPath = await this.findExistingVisualization(graphId);

            if (!pyvisPath) {
                // Generate new visualization
                pyvisPath = await this.generateVisualization(graphId);
            }

            if (pyvisPath) {
                if (loading) loading.style.display = 'none';
                container.innerHTML = `
                    <iframe src="${pyvisPath}" 
                            width="100%" 
                            height="100%" 
                            frameborder="0" 
                            style="border-radius: 8px; background: white;">
                    </iframe>
                `;
                console.log('Graph loaded successfully:', pyvisPath);
            } else {
                throw new Error('Could not generate visualization');
            }
        } catch (error) {
            console.error('Error loading graph:', error);
            this.displayGraphError(container, error.message);
        }
    }

    async findExistingVisualization(graphId) {
        const possiblePaths = [
            `/static/visualizations/${graphId}_graph.html`,
            `/static/visualizations/${graphId}.html`,
            `/static/${graphId}_graph.html`,
            `/instance/view_graph/${graphId}` // Fallback to instance service
        ];

        for (const path of possiblePaths) {
            try {
                const response = await fetch(path, { method: 'HEAD' });
                if (response.ok) {
                    console.log('Found existing visualization at:', path);
                    return path;
                }
            } catch (e) {
                // Continue to next path
            }
        }
        return null;
    }

    async generateVisualization(graphId) {
        try {
            // Try text interface first
            const response = await fetch(`/text_interface/api/generate_pyvis/${graphId}`, {
                method: 'POST'
            });

            if (response.ok) {
                const result = await response.json();
                if (result.success && result.path) {
                    return `/static/${result.path}`;
                }
            }

            // Fallback to instance service direct link
            return `/instance/view_graph/${graphId}`;

        } catch (error) {
            console.error('Error generating visualization:', error);
            // Return instance service as final fallback
            return `/instance/view_graph/${graphId}`;
        }
    }

    async loadSearchVisualization(visualizationPath) {
        const container = document.getElementById('graph-container');
        try {
            const fullPath = `/static/visualizations/${visualizationPath}`;
            container.innerHTML = `
                <iframe src="${fullPath}" 
                        width="100%" 
                        height="100%" 
                        frameborder="0" 
                        style="border-radius: 8px; background: white;">
                </iframe>
            `;
            this.addOutput('Updated graph to show search results.');
        } catch (error) {
            console.error('Error loading search visualization:', error);
            this.addOutput(`<div class="error">Could not load search visualization</div>`);
        }
    }

    displayGraphError(container, errorMessage) {
        container.innerHTML = `
            <div class="graph-error">
                <div class="error-icon">!</div>
                <h3>Graph Visualization Error</h3>
                <p>Unable to load graph visualization</p>
                <p class="error-details">${errorMessage}</p>
                <div class="error-actions">
                    <button onclick="window.buttonInterface.loadGraph()" class="btn btn-primary">Retry</button>
                    <a href="/instance/view_graph/${this.graphData.graph_id}" target="_blank" class="btn btn-secondary">
                        Open in New Tab
                    </a>
                </div>
            </div>
        `;
    }

    resetGraphView() {
        this.loadGraph();
        this.addOutput('Graph view reset.');
    }

    toggleFullscreen() {
        const iframe = document.querySelector('#graph-container iframe');
        if (iframe) {
            window.open(iframe.src, '_blank', 'width=1200,height=800');
            this.addOutput('Opened graph in new window.');
        } else {
            window.open(`/instance/view_graph/${this.graphData.graph_id}`, '_blank');
        }
    }
}

// Initialize when page loads
let buttonInterface;
document.addEventListener('DOMContentLoaded', () => {
    buttonInterface = new ButtonInterface();
    window.buttonInterface = buttonInterface; // Make globally accessible
});