// static/js/propagation/propagation_interface.js

class PropagationInterface {
    constructor() {
        this.data = JSON.parse(document.getElementById('propagation-data').textContent);
        this.propagationResults = null;
        this.filteredResults = null;
        this.layoutResults = null;

        this.setupEventListeners();
        this.populateDropdowns();
        this.loadOriginalGraph();
    }

    setupEventListeners() {
        // Configuration actions
        document.getElementById('run-propagation').addEventListener('click', () => {
            this.runPropagation();
        });

        document.getElementById('reset-config').addEventListener('click', () => {
            this.resetConfiguration();
        });

        // Source node type change
        document.getElementById('source-node-types').addEventListener('change', () => {
            this.updatePropertyDropdown();
        });

        // Results actions
        document.getElementById('show-propagation-modal')?.addEventListener('click', () => {
            this.showPropagationModal();
        });

        document.getElementById('show-filtered-modal')?.addEventListener('click', () => {
            this.showFilteredModal();
        });

        document.getElementById('show-layout-modal')?.addEventListener('click', () => {
            this.showLayoutModal();
        });

        document.getElementById('export-results')?.addEventListener('click', () => {
            this.exportResults();
        });

        // Graph controls
        document.getElementById('toggle-original').addEventListener('click', () => {
            this.showOriginalGraph();
        });

        document.getElementById('toggle-propagation').addEventListener('click', () => {
            this.showPropagationGraph();
        });

        document.getElementById('fullscreen-graph').addEventListener('click', () => {
            this.toggleFullscreen();
        });

        // Modal close
        this.setupModalClose();
    }

    populateDropdowns() {
        const metadata = this.data.metadata;

        // Populate node types
        const nodeTypesSelect = document.getElementById('source-node-types');
        Object.entries(metadata.node_types).forEach(([type, count]) => {
            const option = document.createElement('option');
            option.value = type;
            option.textContent = `${type} (${count} nodes)`;
            nodeTypesSelect.appendChild(option);
        });

         // Populate exclude node types
        const excludeNodeTypesSelect = document.getElementById('exclude-node-types');
        Object.entries(metadata.node_types).forEach(([type, count]) => {
            const option = document.createElement('option');
            option.value = type;
            option.textContent = `${type} (${count} nodes)`;
            excludeNodeTypesSelect.appendChild(option);
        });

        // Populate exclude edge types
        const excludeEdgeTypesSelect = document.getElementById('exclude-edge-types');
        metadata.edge_types.forEach(type => {
            const option = document.createElement('option');
            option.value = type;
            option.textContent = type;
            excludeEdgeTypesSelect.appendChild(option);
        });
    }

    updatePropertyDropdown() {
        const selectedTypes = Array.from(document.getElementById('source-node-types').selectedOptions)
            .map(option => option.value);

        const propertySelect = document.getElementById('source-property');
        propertySelect.innerHTML = '<option value="">Select property...</option>';

        const allProperties = new Set();
        selectedTypes.forEach(type => {
            if (this.data.metadata.properties_by_type[type]) {
                this.data.metadata.properties_by_type[type].forEach(prop => {
                    allProperties.add(prop);
                });
            }
        });

        allProperties.forEach(prop => {
            const option = document.createElement('option');
            option.value = prop;
            option.textContent = prop;
            propertySelect.appendChild(option);
        });
    }

async runPropagation() {
    const config = this.getConfiguration();
    if (!this.validateConfiguration(config)) {
        return;
    }

    this.setStatus('running', 'Running propagation...');
    this.propagationStartTime = Date.now();

    try {
        console.log('DEBUG: Sending request with config:', config);

        // First, get metadata to know how many chunks we need
        const metadataConfig = { ...config, metadata_only: true };
        const metadataResponse = await fetch(this.data.api_endpoints.full_workflow, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(metadataConfig)
        });

        if (!metadataResponse.ok) {
            throw new Error(`Metadata request failed: ${metadataResponse.status}`);
        }

        const metadata = await metadataResponse.json();
        if (!metadata.success) {
            throw new Error(metadata.error || 'Failed to get metadata');
        }

        console.log(`DEBUG: Need to load ${metadata.metadata.total_chunks} chunks`);
        this.addStatusMessage('info', `Loading ${metadata.metadata.total_nodes} nodes in ${metadata.metadata.total_chunks} chunks...`);

        // Initialize results structure
        this.propagationResults = {
            propagated_values: {},
            gmm_data: {},
            source_nodes: metadata.metadata.source_nodes,
            value_range: metadata.metadata.value_range,
            method: metadata.metadata.method
        };

        // Load chunks sequentially
        for (let chunkIndex = 0; chunkIndex < metadata.metadata.total_chunks; chunkIndex++) {
            console.log(`DEBUG: Loading chunk ${chunkIndex + 1}/${metadata.metadata.total_chunks}`);

            const chunkConfig = {
                ...config,
                chunk_index: chunkIndex,
                chunk_size: 1000 // Adjust based on what works
            };

            let chunkResponse;
            let chunkResult;

            try {
                chunkResponse = await fetch(this.data.api_endpoints.full_workflow, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify(chunkConfig)
                });

                console.log(`DEBUG: Chunk ${chunkIndex + 1} response status:`, chunkResponse.status);
                console.log(`DEBUG: Chunk ${chunkIndex + 1} response headers:`, [...chunkResponse.headers.entries()]);

                if (!chunkResponse.ok) {
                    const errorText = await chunkResponse.text();
                    console.error(`DEBUG: Chunk ${chunkIndex + 1} error response:`, errorText);
                    throw new Error(`Chunk ${chunkIndex + 1} request failed: ${chunkResponse.status} - ${errorText}`);
                }

                // Get response as text first to debug
                const responseText = await chunkResponse.text();
                console.log(`DEBUG: Chunk ${chunkIndex + 1} response length:`, responseText.length);
                console.log(`DEBUG: Chunk ${chunkIndex + 1} response start:`, responseText.substring(0, 200));
                console.log(`DEBUG: Chunk ${chunkIndex + 1} response end:`, responseText.substring(responseText.length - 200));

                if (!responseText || responseText.trim() === '') {
                    throw new Error(`Chunk ${chunkIndex + 1} returned empty response`);
                }

                try {
                    chunkResult = JSON.parse(responseText);
                } catch (jsonError) {
                    console.error(`DEBUG: Chunk ${chunkIndex + 1} JSON parse error:`, jsonError);
                    console.error(`DEBUG: Chunk ${chunkIndex + 1} response preview:`, responseText.substring(0, 1000));

                    // Try to find where JSON breaks
                    let lastValidJson = '';
                    for (let i = responseText.length; i > 0; i -= 1000) {
                        try {
                            const testJson = responseText.substring(0, i);
                            JSON.parse(testJson + '}');
                            lastValidJson = testJson;
                            break;
                        } catch (e) {
                            continue;
                        }
                    }
                    console.error(`DEBUG: Chunk ${chunkIndex + 1} last valid JSON length:`, lastValidJson.length);
                    throw new Error(`Chunk ${chunkIndex + 1} JSON parsing failed at position ~${lastValidJson.length}. Response may be truncated.`);
                }

            } catch (fetchError) {
                console.error(`DEBUG: Error loading chunk ${chunkIndex + 1}:`, fetchError);
                throw new Error(`Failed to load chunk ${chunkIndex + 1}: ${fetchError.message}`);
            }

            if (!chunkResult.success) {
                throw new Error(chunkResult.error || `Chunk ${chunkIndex + 1} failed`);
            }

            console.log(`DEBUG: Chunk ${chunkIndex + 1} loaded successfully`);
            console.log(`DEBUG: Chunk ${chunkIndex + 1} data keys:`, Object.keys(chunkResult.chunk_data || {}));

            // Merge chunk data
            if (chunkResult.chunk_data) {
                if (chunkResult.chunk_data.propagated_values) {
                    Object.assign(this.propagationResults.propagated_values, chunkResult.chunk_data.propagated_values);
                    console.log(`DEBUG: Added ${Object.keys(chunkResult.chunk_data.propagated_values).length} propagated values from chunk ${chunkIndex + 1}`);
                }
                if (chunkResult.chunk_data.gmm_data) {
                    Object.assign(this.propagationResults.gmm_data, chunkResult.chunk_data.gmm_data);
                    console.log(`DEBUG: Added ${Object.keys(chunkResult.chunk_data.gmm_data).length} GMM data entries from chunk ${chunkIndex + 1}`);
                }
            }

            // Update progress
            const progress = ((chunkIndex + 1) / metadata.metadata.total_chunks) * 100;
            this.setStatus('running', `Loading data... ${progress.toFixed(0)}%`);

            // If this is the last chunk, it might contain additional results
            if (chunkIndex === metadata.metadata.total_chunks - 1) {
                console.log(`DEBUG: Processing final chunk ${chunkIndex + 1} additional data`);
                if (chunkResult.filtered_results) {
                    this.filteredResults = chunkResult.filtered_results;
                    console.log('DEBUG: Added filtered results from final chunk');
                }
                if (chunkResult.layout) {
                    this.layoutResults = chunkResult.layout;
                    console.log('DEBUG: Added layout results from final chunk');
                }
            }

            // Small delay to prevent UI blocking
            await new Promise(resolve => setTimeout(resolve, 10));
        }

        console.log('DEBUG: Successfully loaded all chunks');
        console.log(`DEBUG: Total propagated values: ${Object.keys(this.propagationResults.propagated_values).length}`);
        console.log(`DEBUG: Total GMM data: ${Object.keys(this.propagationResults.gmm_data).length}`);

        this.displayResults({
            success: true,
            propagation_results: this.propagationResults,
            filtered_results: this.filteredResults,
            layout: this.layoutResults
        });

        this.setStatus('complete', 'Propagation complete!');

        // Generate visualization
        if (this.propagationResults) {
            await this.generatePropagationVisualization({
                propagation_results: this.propagationResults,
                filtered_results: this.filteredResults,
                layout: this.layoutResults
            });
        }

    } catch (error) {
        console.error('Propagation error:', error);
        this.setStatus('error', `Error: ${error.message}`);
        this.addStatusMessage('error', `Propagation failed: ${error.message}`);
    }
}

getConfiguration() {
    const sourceNodeTypes = Array.from(document.getElementById('source-node-types').selectedOptions)
        .map(option => option.value);

    const sourceProperty = document.getElementById('source-property').value;
    const method = document.getElementById('propagation-method').value;
    const graphDirection = document.getElementById('graph-direction').value;

    const minValue = document.getElementById('min-value').value;
    const maxValue = document.getElementById('max-value').value;
    const layoutType = document.getElementById('layout-type').value;

    const excludeNodeTypes = Array.from(document.getElementById('exclude-node-types').selectedOptions)
        .map(option => option.value);

    const excludeEdgeTypes = Array.from(document.getElementById('exclude-edge-types').selectedOptions)
        .map(option => option.value);

    const config = {
        graph_id: this.data.graph_id,
        source_node_types: sourceNodeTypes,
        source_property: sourceProperty,
        method: method,
        graph_direction: graphDirection,  // NEW
        calculate_layout: true,
        layout_config: {
            layout_type: layoutType,
            base_radius: 150
        },
        include_neighbors: true,
        exclude_node_types: excludeNodeTypes,  // NEW
        exclude_edge_types: excludeEdgeTypes,  // NEW
    };

    // Add range filter if specified
    if (minValue !== '' && maxValue !== '') {
        config.min_value = parseFloat(minValue);
        config.max_value = parseFloat(maxValue);
    }

    return config;
}

   validateConfiguration(config) {
       if (!config.source_node_types.length) {
           this.addStatusMessage('error', 'Please select at least one source node type');
           return false;
       }

       if (!config.source_property) {
           this.addStatusMessage('error', 'Please select a source property');
           return false;
       }

       if (config.min_value !== undefined && config.max_value !== undefined) {
           if (config.min_value >= config.max_value) {
               this.addStatusMessage('error', 'Min value must be less than max value');
               return false;
           }
       }

       return true;
   }

displayResults(result) {
    // Show results section
    document.getElementById('results-section').style.display = 'block';
    // In displayResults method, add this line:
    document.getElementById('propagation-legend').style.display = 'block';

    // Update summary
    const summary = document.getElementById('results-summary');
    const propagatedCount = Object.keys(result.propagation_results.propagated_values).length;
    const filteredCount = result.filtered_results ? result.filtered_results.nodes_in_range.length : 0;

    summary.innerHTML = `
        <span><strong>${propagatedCount}</strong> nodes with propagated values</span>
        ${result.filtered_results ? `<span><strong>${filteredCount}</strong> nodes in range</span>` : ''}
        <span>Method: <strong>${result.propagation_results.method}</strong></span>
    `;

    // Update button summaries
    document.getElementById('propagation-summary').textContent =
        `${propagatedCount} propagated values`;

    if (result.filtered_results) {
        document.getElementById('filtered-summary').textContent =
            `${filteredCount} nodes in range`;
    }

    if (result.layout) {
        document.getElementById('layout-summary').textContent =
            `${Object.keys(result.layout.positions).length} positioned nodes`;
    }

    // Enable propagation graph button
    document.getElementById('toggle-propagation').disabled = false;

    // Show exclusion info
    const config = this.getConfiguration();
    let exclusionInfo = '';

    if (config.exclude_node_types && config.exclude_node_types.length > 0) {
        exclusionInfo += `Excluded node types: ${config.exclude_node_types.join(', ')}. `;
    }

    if (config.exclude_edge_types && config.exclude_edge_types.length > 0) {
        exclusionInfo += `Excluded edge types: ${config.exclude_edge_types.join(', ')}.`;
    }

    if (exclusionInfo) {
        this.addStatusMessage('info', `Graph filtered: ${exclusionInfo}`);
    }

    // Add success message
    this.addStatusMessage('success',
        `Propagation completed successfully! Found ${propagatedCount} nodes with propagated values.`);
}

async generatePropagationVisualization(result) {
    try {
        console.log('DEBUG: Starting visualization generation...');
        console.log('DEBUG: API endpoint:', this.data.api_endpoints.visualization);
        console.log('DEBUG: Result data:', result);

        const visResponse = await fetch(this.data.api_endpoints.visualization, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                subgraph: result.filtered_subgraph || result.propagation_results,
                propagation_results: result.propagation_results,
                layout_positions: result.layout ? result.layout.positions : null
            })
        });

        console.log('DEBUG: Visualization response status:', visResponse.status);

        if (visResponse.ok) {
            const visResult = await visResponse.json();
            console.log('DEBUG: Visualization result:', visResult);

            if (visResult.success) {
                this.propagationVisualizationPath = `/static/visualizations/${visResult.visualization_path}`;
                console.log('DEBUG: Set visualization path:', this.propagationVisualizationPath);
                this.addStatusMessage('info', 'Propagation visualization generated successfully');
            } else {
                console.log('DEBUG: Visualization failed:', visResult.error);
                this.addStatusMessage('warning', `Visualization failed: ${visResult.error}`);
            }
        } else {
            const errorText = await visResponse.text();
            console.log('DEBUG: Visualization request failed:', errorText);
            this.addStatusMessage('warning', `Visualization request failed: ${visResponse.status}`);
        }
    } catch (error) {
        console.error('Visualization generation error:', error);
        this.addStatusMessage('warning', 'Could not generate propagation visualization');
    }
}

   showPropagationModal() {
       if (!this.propagationResults) return;

       const content = this.formatPropagationResults(this.propagationResults);
       this.showModal('Propagation Analysis', content);
   }

   showFilteredModal() {
       if (!this.filteredResults) return;

       const content = this.formatFilteredResults(this.filteredResults);
       this.showModal('Filtered Results', content);
   }

   showLayoutModal() {
       if (!this.layoutResults) return;

       const content = this.formatLayoutResults(this.layoutResults);
       this.showModal('Layout Analysis', content);
   }

   formatPropagationResults(results) {
       const valueCount = Object.keys(results.propagated_values).length;
       const sourceCount = results.source_nodes.length;
       const valueRange = results.value_range;

       let html = `
           <div class="results-overview">
               <div class="overview-grid">
                   <div class="overview-card">
                       <div class="overview-number">${valueCount}</div>
                       <div class="overview-label">Propagated Values</div>
                   </div>
                   <div class="overview-card">
                       <div class="overview-number">${sourceCount}</div>
                       <div class="overview-label">Source Nodes</div>
                   </div>
                   <div class="overview-card">
                       <div class="overview-number">${valueRange[1] - valueRange[0]}</div>
                       <div class="overview-label">Value Range</div>
                   </div>
               </div>
           </div>

           <div class="propagation-details">
               <h4>Source Nodes</h4>
               <div class="source-nodes-grid">
       `;

       results.source_nodes.forEach(source => {
           html += `
               <div class="source-node-card">
                   <div class="source-id">${source.node_id}</div>
                   <div class="source-value">Value: ${source.raw_value}</div>
               </div>
           `;
       });

       html += `
               </div>
               
               <h4>Propagated Values (Top 20)</h4>
               <div class="propagated-values-list">
       `;

       // Show top 20 propagated values
       const sortedValues = Object.entries(results.propagated_values)
           .sort(([,a], [,b]) => b - a)
           .slice(0, 20);

       sortedValues.forEach(([nodeId, value]) => {
           const gmm = results.gmm_data[nodeId];
           const confidence = gmm && gmm.total_evidence ? gmm.total_evidence.toFixed(3) : 'N/A';
           const components = gmm ? gmm.n_components : 0;

           html += `
               <div class="propagated-value-item">
                   <div class="value-header">
                       <span class="value-node-id">${nodeId}</span>
                       <span class="value-amount">${value.toFixed(3)}</span>
                   </div>
                   <div class="value-details">
                       <span>Confidence: ${confidence}</span>
                       <span>Components: ${components}</span>
                       <span>Type: ${gmm ? gmm.type : 'unknown'}</span>
                   </div>
               </div>
           `;
       });

       html += `</div></div>`;
       return html;
   }

   formatFilteredResults(results) {
       const inRangeCount = results.nodes_in_range.length;
       const outRangeCount = results.nodes_out_of_range.length;
       const filter = results.range_filter;

       let html = `
           <div class="filter-overview">
               <div class="filter-summary">
                   <h4>Range Filter: ${filter.min} to ${filter.max}</h4>
                   <div class="filter-stats">
                       <div class="filter-stat success">
                           <span class="stat-number">${inRangeCount}</span>
                           <span class="stat-label">Nodes in Range</span>
                       </div>
                       <div class="filter-stat muted">
                           <span class="stat-number">${outRangeCount}</span>
                           <span class="stat-label">Nodes out of Range</span>
                       </div>
                   </div>
               </div>
           </div>

           <div class="filtered-details">
               <h4>Nodes in Range</h4>
               <div class="filtered-nodes-grid">
       `;

       results.nodes_in_range.forEach(node => {
           const valueText = node.value !== null ? node.value.toFixed(3) : 'N/A';
           const typeClass = node.gmm_type === 'source' ? 'source' :
                           node.gmm_type === 'inferred' ? 'inferred' : 'isolated';

           html += `
               <div class="filtered-node-card ${typeClass}">
                   <div class="node-header">
                       <span class="node-id">${node.node_id}</span>
                       <span class="node-type-badge ${typeClass}">${node.gmm_type}</span>
                   </div>
                   <div class="node-value">Value: ${valueText}</div>
                   ${node.n_components > 0 ? `<div class="node-components">Components: ${node.n_components}</div>` : ''}
               </div>
           `;
       });

       html += `</div></div>`;
       return html;
   }

   formatLayoutResults(results) {
       const positionCount = Object.keys(results.positions).length;
       const layoutType = results.layout_type;

       let html = `
           <div class="layout-overview">
               <h4>Layout Analysis: ${layoutType}</h4>
               <p>${positionCount} nodes positioned based on similarity analysis</p>
           </div>

           <div class="layout-details">
               <h4>Position Data (Sample)</h4>
               <div class="position-list">
       `;

       // Show first 15 positions
       const positions = Object.entries(results.positions).slice(0, 15);

       positions.forEach(([nodeId, pos]) => {
           html += `
               <div class="position-item">
                   <div class="position-header">
                       <span class="position-node-id">${nodeId}</span>
                   </div>
                   <div class="position-coords">
                       X: ${pos.x.toFixed(2)}, Y: ${pos.y.toFixed(2)}
                       ${pos.radius ? `, R: ${pos.radius.toFixed(2)}` : ''}
                       ${pos.centrality ? `, Centrality: ${pos.centrality.toFixed(3)}` : ''}
                   </div>
               </div>
           `;
       });

       if (Object.keys(results.positions).length > 15) {
           html += `<div class="more-positions">... and ${Object.keys(results.positions).length - 15} more positions</div>`;
       }

       html += `</div></div>`;
       return html;
   }

   exportResults() {
       if (!this.propagationResults) {
           this.addStatusMessage('warning', 'No results to export');
           return;
       }

       const exportData = {
           graph_id: this.data.graph_id,
           timestamp: new Date().toISOString(),
           propagation_results: this.propagationResults,
           filtered_results: this.filteredResults,
           layout_results: this.layoutResults,
           configuration: this.getConfiguration()
       };

       const blob = new Blob([JSON.stringify(exportData, null, 2)], {
           type: 'application/json'
       });

       const url = URL.createObjectURL(blob);
       const a = document.createElement('a');
       a.href = url;
       a.download = `propagation_results_${this.data.graph_id}_${Date.now()}.json`;
       document.body.appendChild(a);
       a.click();
       document.body.removeChild(a);
       URL.revokeObjectURL(url);

       this.addStatusMessage('success', 'Results exported successfully');
   }

   showOriginalGraph() {
       this.loadOriginalGraph();
       document.getElementById('toggle-original').style.background = '#28a745';
       document.getElementById('toggle-propagation').style.background = '#3498db';
   }

showPropagationGraph() {
    if (this.propagationVisualizationPath) {
        this.loadGraph(this.propagationVisualizationPath);
        document.getElementById('toggle-propagation').style.background = '#28a745';
        document.getElementById('toggle-original').style.background = '#3498db';

        // Add message about interactive features
        this.addStatusMessage('info', 'Click on any node to see its signal plot and propagation details!');
    } else {
        this.addStatusMessage('warning', 'No propagation visualization available');
    }
}

   async loadOriginalGraph() {
       const container = document.getElementById('graph-container');
       this.showGraphLoading('Loading original graph...');

       try {
           // Try to find existing visualization
           const graphId = this.data.graph_id;
           const possiblePaths = [
               `/static/visualizations/${graphId}.html`,
               `/instance/view_graph/${graphId}`
           ];

           let loadPath = null;
           for (const path of possiblePaths) {
               try {
                   const response = await fetch(path, { method: 'HEAD' });
                   if (response.ok) {
                       loadPath = path;
                       break;
                   }
               } catch (e) {
                   // Continue to next path
               }
           }

           if (!loadPath) {
               loadPath = `/instance/view_graph/${graphId}`;
           }

           this.loadGraph(loadPath);

       } catch (error) {
           console.error('Error loading original graph:', error);
           this.displayGraphError(container, error.message);
       }
   }

   loadGraph(path) {
       const container = document.getElementById('graph-container');
       container.innerHTML = `
           <iframe src="${path}" 
                   width="100%" 
                   height="100%" 
                   frameborder="0" 
                   style="border-radius: 8px; background: white;">
           </iframe>
       `;
   }

   showGraphLoading(message) {
       const container = document.getElementById('graph-container');
       container.innerHTML = `
           <div class="graph-loading">
               <div class="loading-spinner"></div>
               <div class="loading-text">${message}</div>
           </div>
       `;
   }

   displayGraphError(container, message) {
       container.innerHTML = `
           <div class="graph-error">
               <div class="error-icon">!</div>
               <h3>Graph Loading Error</h3>
               <p>${message}</p>
               <button onclick="window.propagationInterface.loadOriginalGraph()" class="btn btn-primary">
                   Retry
               </button>
           </div>
       `;
   }

   toggleFullscreen() {
       const iframe = document.querySelector('#graph-container iframe');
       if (iframe) {
           window.open(iframe.src, '_blank', 'width=1200,height=800');
       }
   }

   resetConfiguration() {
       document.getElementById('source-node-types').selectedIndex = -1;
       document.getElementById('source-property').selectedIndex = 0;
       document.getElementById('propagation-method').value = 'matrix';
       document.getElementById('exclude-node-types').selectedIndex = -1;
       document.getElementById('exclude-edge-types').selectedIndex = -1;

       document.getElementById('min-value').value = '';
       document.getElementById('max-value').value = '';
       document.getElementById('layout-type').value = 'polar';

       // Clear results
       document.getElementById('results-section').style.display = 'none';
       this.propagationResults = null;
       this.filteredResults = null;
       this.layoutResults = null;

       // Reset status
       this.setStatus('ready', 'Ready');
       this.clearStatusMessages();

       // Disable propagation graph button
       document.getElementById('toggle-propagation').disabled = true;
       this.showOriginalGraph();
   }

   setStatus(type, message) {
       const indicator = document.getElementById('status-indicator');
       const dot = indicator.querySelector('.indicator-dot');

       dot.className = `indicator-dot ${type}`;
       indicator.childNodes[2].textContent = message; // Update text node
   }

   addStatusMessage(type, message) {
       const content = document.getElementById('status-content');

       // Remove welcome message if present
       const welcome = content.querySelector('.status-welcome');
       if (welcome) {
           welcome.remove();
       }

       const messageDiv = document.createElement('div');
       messageDiv.className = `status-message ${type}`;
       messageDiv.innerHTML = `
           <div class="message-time">${new Date().toLocaleTimeString()}</div>
           <div class="message-text">${message}</div>
       `;

       content.appendChild(messageDiv);
       content.scrollTop = content.scrollHeight;
   }

   clearStatusMessages() {
       const content = document.getElementById('status-content');
       content.innerHTML = `
           <div class="status-welcome">
               <div class="welcome-header">Configure & Run</div>
               <p>Set your propagation parameters above and click "Run Propagation" to begin analysis.</p>
               <div class="quick-tips">
                   <h4>Quick Tips:</h4>
                   <ul>
                       <li>Choose node types that contain your source property</li>
                       <li>Use range filtering to focus on specific value ranges</li>
                       <li>Try different methods for comparison</li>
                   </ul>
               </div>
           </div>
       `;
   }

   showProgress(percentage) {
       let progressContainer = document.querySelector('.progress-container');
       if (!progressContainer) {
           progressContainer = document.createElement('div');
           progressContainer.className = 'progress-container';
           progressContainer.innerHTML = `
               <div class="progress-bar">
                   <div class="progress-fill"></div>
               </div>
           `;
           document.getElementById('status-content').appendChild(progressContainer);
       }

       const fill = progressContainer.querySelector('.progress-fill');
       fill.style.width = `${percentage}%`;
   }

   hideProgress() {
       const progressContainer = document.querySelector('.progress-container');
       if (progressContainer) {
           setTimeout(() => progressContainer.remove(), 1000);
       }
   }

   showModal(title, content) {
       const modal = document.getElementById('propagation-analysis-modal');
       const modalTitle = document.getElementById('modal-title');
       const modalBody = document.getElementById('modal-body');

       modalTitle.textContent = title;
       modalBody.innerHTML = content;
       modal.style.display = 'block';
       document.body.style.overflow = 'hidden';
   }

   hideModal() {
       const modal = document.getElementById('propagation-analysis-modal');
       modal.style.display = 'none';
       document.body.style.overflow = 'auto';
   }

   setupModalClose() {
       const modal = document.getElementById('propagation-analysis-modal');
       const closeBtn = modal.querySelector('.close');

       closeBtn.addEventListener('click', () => {
           this.hideModal();
       });

       window.addEventListener('click', (event) => {
           if (event.target === modal) {
               this.hideModal();
           }
       });
   }
}

// Signal analysis window creation - called from the propagation visualization
function createSignalAnalysisWindow(signalWindow, node, nodeId, valueRange) {
    const doc = signalWindow.document;

    doc.open();
    doc.write(`
    <!DOCTYPE html>
    <html>
    <head>
        <title>Signal Analysis: ${nodeId}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f8f9fa; }
            .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            .header { border-bottom: 2px solid #007bff; padding-bottom: 15px; margin-bottom: 20px; }
            .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
            .info-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }
            .info-card h4 { margin: 0 0 10px 0; color: #2c3e50; }
            .component-item { background: white; margin-bottom: 8px; padding: 10px; border-radius: 4px; border-left: 3px solid #28a745; }
            #signal-plot { width: 100%; height: 500px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔬 Signal Analysis: ${nodeId}</h1>
                <p>Interactive signal component visualization</p>
            </div>

            <div class="info-grid">
                <div class="info-card">
                    <h4>Node Information</h4>
                    <p><strong>ID:</strong> ${nodeId}</p>
                    <p><strong>Type:</strong> ${node.type}</p>
                    <p><strong>Source:</strong> ${node.is_source ? 'Yes' : 'No'}</p>
                    <p><strong>Propagated Value:</strong> ${node.propagated_value !== null ? node.propagated_value.toFixed(3) : 'N/A'}</p>
                </div>

                <div class="info-card">
                    <h4>Signal Components</h4>
                    <div id="components-list">
                        ${formatComponents(node.gmm_data)}
                    </div>
                </div>
            </div>

            <div id="signal-plot"></div>
        </div>

        <script>
            const nodeData = ${JSON.stringify(node)};
            const valueRange = ${JSON.stringify(valueRange)};

            function formatComponents(gmmData) {
                if (!gmmData || !gmmData.components || gmmData.components.length === 0) {
                    return '<p>No signal components available</p>';
                }

                let html = '';
                gmmData.components.forEach(function(comp, i) {
                    const weight = comp[0];
                    const mean = comp[1]; 
                    const std = comp[2];
                    html += '<div class="component-item">';
                    html += '<strong>Component ' + (i + 1) + '</strong><br>';
                    html += 'Weight: ' + weight.toFixed(3) + '<br>';
                    html += 'Mean: ' + mean.toFixed(3) + '<br>';
                    html += 'Std: ' + std.toFixed(3);
                    html += '</div>';
                });
                return html;
            }

            // Generate the plot
            function generateSignalPlot() {
                const plotDiv = document.getElementById('signal-plot');

                if (!nodeData.gmm_data || !nodeData.gmm_data.components || nodeData.gmm_data.components.length === 0) {
                    plotDiv.innerHTML = '<div style="text-align: center; padding: 50px; color: #666;"><h3>No Signal Data Available</h3><p>This node has no propagated signal components to display.</p></div>';
                    return;
                }

                // Create x-range for plotting
                const xRange = [];
                const step = (valueRange[1] - valueRange[0]) / 300;
                for (let x = valueRange[0]; x <= valueRange[1]; x += step) {
                    xRange.push(x);
                }

                const traces = [];
                const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd', '#98d8c8'];

                // Plot individual components
                nodeData.gmm_data.components.forEach(function(comp, i) {
                    const weight = comp[0];
                    const mean = comp[1];
                    const std = comp[2];

                    // Denormalize for actual values
                    const actualMean = mean * (valueRange[1] - valueRange[0]) + valueRange[0];
                    const actualStd = std * (valueRange[1] - valueRange[0]);

                    const yValues = xRange.map(function(x) {
                        const gaussian = weight * (1 / (actualStd * Math.sqrt(2 * Math.PI))) * 
                                       Math.exp(-0.5 * Math.pow((x - actualMean) / actualStd, 2));
                        return gaussian;
                    });

                    traces.push({
                        x: xRange,
                        y: yValues,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Component ' + (i + 1) + ' (w=' + weight.toFixed(2) + ')',
                        line: { color: colors[i % colors.length], width: 2 },
                        opacity: 0.8
                    });
                });

                // Plot combined signal
                const combinedY = xRange.map(function(x) {
                    let total = 0;
                    nodeData.gmm_data.components.forEach(function(comp) {
                        const weight = comp[0];
                        const mean = comp[1];
                        const std = comp[2];
                        const actualMean = mean * (valueRange[1] - valueRange[0]) + valueRange[0];
                        const actualStd = std * (valueRange[1] - valueRange[0]);
                        const gaussian = weight * (1 / (actualStd * Math.sqrt(2 * Math.PI))) * 
                                       Math.exp(-0.5 * Math.pow((x - actualMean) / actualStd, 2));
                        total += gaussian;
                    });
                    return total;
                });

                traces.push({
                    x: xRange,
                    y: combinedY,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Combined Signal',
                    line: { width: 4, color: 'black' },
                    opacity: 1.0
                });

                // Add vertical line for propagated value
                if (nodeData.propagated_value !== null) {
                    const maxY = Math.max.apply(Math, combinedY);
                    traces.push({
                        x: [nodeData.propagated_value, nodeData.propagated_value],
                        y: [0, maxY * 1.1],
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Final Value',
                        line: { color: 'red', dash: 'dash', width: 3 },
                        opacity: 0.8
                    });
                }

                const layout = {
                    title: {
                        text: 'Signal Distribution for ' + nodeData.id,
                        font: { size: 18 }
                    },
                    xaxis: { 
                        title: 'Value',
                        gridcolor: '#e1e5e9',
                        showgrid: true
                    },
                    yaxis: { 
                        title: 'Probability Density',
                        gridcolor: '#e1e5e9',
                        showgrid: true
                    },
                    showlegend: true,
                    legend: { x: 0.02, y: 0.98 },
                    margin: { t: 60, b: 60, l: 80, r: 60 },
                    plot_bgcolor: '#fafafa',
                    paper_bgcolor: 'white'
                };

                const config = {
                    responsive: true,
                    displayModeBar: true,
                    toImageButtonOptions: {
                        format: 'png',
                        filename: 'signal_' + nodeData.id,
                        height: 500,
                        width: 900,
                        scale: 1
                    }
                };

                Plotly.newPlot(plotDiv, traces, layout, config);
            }

            // Generate plot when page loads
            document.addEventListener('DOMContentLoaded', generateSignalPlot);
        </script>
    </body>
    </html>
    `);
    doc.close();

    // Update components list after document is ready
    setTimeout(() => {
        const componentsList = signalWindow.document.getElementById('components-list');
        if (componentsList) {
            componentsList.innerHTML = formatComponents(node.gmm_data);
        }
    }, 100);
}

function formatComponents(gmmData) {
    if (!gmmData || !gmmData.components || gmmData.components.length === 0) {
        return '<p>No signal components available</p>';
    }

    let html = '';
    gmmData.components.forEach(function(comp, i) {
        const weight = comp[0];
        const mean = comp[1];
        const std = comp[2];
        html += '<div class="component-item">';
        html += '<strong>Component ' + (i + 1) + '</strong><br>';
        html += 'Weight: ' + weight.toFixed(3) + '<br>';
        html += 'Mean: ' + mean.toFixed(3) + '<br>';
        html += 'Std: ' + std.toFixed(3);
        html += '</div>';
    });
    return html;
}

// Make functions globally available
window.createSignalAnalysisWindow = createSignalAnalysisWindow;
window.formatComponents = formatComponents;

// Initialize when page loads
let propagationInterface;
document.addEventListener('DOMContentLoaded', () => {
   propagationInterface = new PropagationInterface();
   window.propagationInterface = propagationInterface;
});