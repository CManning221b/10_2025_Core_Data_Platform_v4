// static/js/visualisation/channel_button.js
class ChannelButton {
    constructor(data, color, nameKey, returnKey, size = null) {
        this.data = data;
        this.color = color;
        this.nameKey = nameKey;
        this.returnKey = returnKey;
        this.size = size;
    }

    createButton() {
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'circular-button-container';

        const button = document.createElement('button');
        button.textContent = this.data[this.nameKey];
        button.className = 'circular-button';
        button.dataset.name = this.data[this.nameKey];
        button.dataset.value = this.data[this.returnKey];

        // Set the background color using CSS custom property
        button.style.setProperty('--main-color', this.color);
        button.style.backgroundColor = this.color;

        if (this.size !== null) {
            const containerSize = this.size * 0.9;
            button.style.width = `${containerSize}px`;
            button.style.height = `${containerSize}px`;
            const fontSize = Math.max(8, containerSize * 0.25);
            button.style.fontSize = `${fontSize}px`;
        }

        // Add click handler with rich channel info
        button.addEventListener('click', (e) => {
            e.preventDefault();
            console.log('Channel clicked:', this.data[this.nameKey], this.data);
            this.showChannelModal();
        });

        // Enhanced hover effects
        button.addEventListener('mouseenter', () => {
            button.style.transform = 'scale(1.2) translateZ(0)';
            button.style.boxShadow = '0 8px 25px rgba(0,0,0,0.3)';
            button.style.zIndex = '100';
        });

        button.addEventListener('mouseleave', () => {
            button.style.transform = 'scale(1)';
            button.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
            button.style.zIndex = '1';
        });

        buttonContainer.appendChild(button);
        return buttonContainer;
    }

    showChannelModal() {
        // Remove existing modal if any
        const existingModal = document.getElementById('channel-modal');
        if (existingModal) {
            existingModal.remove();
        }

        // Create modal
        const modal = document.createElement('div');
        modal.id = 'channel-modal';
        modal.className = 'channel-modal';
        modal.style.display = 'block';

        const modalContent = document.createElement('div');
        modalContent.className = 'channel-modal-content';

        // Create close button
        const closeButton = document.createElement('span');
        closeButton.className = 'close-modal';
        closeButton.innerHTML = '&times;';
        closeButton.onclick = () => modal.remove();

        // Create content
        const title = document.createElement('h2');
        title.textContent = `Channel ${this.data[this.nameKey]} Details`;

        const infoGrid = document.createElement('div');
        infoGrid.className = 'channel-info-grid';

        // Basic info
        const basicInfo = [
            { label: 'Channel ID', value: this.data[this.nameKey] },
            { label: 'Status', value: this.data.status || 'unknown' },
            { label: 'Measurements', value: this.data.measurement_count || 0 },
            { label: 'Node Type', value: this.data.node_type || 'N/A' }
        ];

        if (this.data.last_measurement) {
            basicInfo.push({
                label: 'Last Measurement',
                value: new Date(this.data.last_measurement).toLocaleString()
            });
        }

        if (this.data.parent_folder) {
            basicInfo.push({ label: 'Parent Folder', value: this.data.parent_folder });
        }

        // Add basic info items
        basicInfo.forEach(item => {
            const infoItem = document.createElement('div');
            infoItem.className = 'info-item';

            const label = document.createElement('div');
            label.className = 'info-label';
            label.textContent = item.label;

            const value = document.createElement('div');
            value.className = 'info-value';
            value.textContent = item.value;

            infoItem.appendChild(label);
            infoItem.appendChild(value);
            infoGrid.appendChild(infoItem);
        });

        // Temporal data section
        if (this.data.temporal_data && this.data.temporal_data.length > 0) {
            const temporalSection = document.createElement('div');
            temporalSection.innerHTML = `
                <h3 style="margin-top: 25px; color: #667eea;">Temporal Data (${this.data.temporal_data.length} entries)</h3>
            `;

            const temporalList = document.createElement('div');
            temporalList.style.maxHeight = '200px';
            temporalList.style.overflowY = 'auto';
            temporalList.style.background = '#f8f9fa';
            temporalList.style.padding = '15px';
            temporalList.style.borderRadius = '10px';
            temporalList.style.marginTop = '10px';

            this.data.temporal_data.forEach((entry, index) => {
                const entryDiv = document.createElement('div');
                entryDiv.style.padding = '8px';
                entryDiv.style.borderBottom = '1px solid #dee2e6';
                entryDiv.innerHTML = `
                    <strong>${index + 1}.</strong> ${entry.datetime || 'No datetime'} 
                    <small style="color: #666;">(ID: ${entry.timestamp_id})</small>
                `;
                temporalList.appendChild(entryDiv);
            });

            temporalSection.appendChild(temporalList);
            modalContent.appendChild(temporalSection);
        }

        // Raw data section (collapsible)
        if (this.data.raw_node_data) {
            const rawSection = document.createElement('details');
            rawSection.style.marginTop = '20px';

            const summary = document.createElement('summary');
            summary.textContent = 'Raw Node Data';
            summary.style.cursor = 'pointer';
            summary.style.fontWeight = 'bold';
            summary.style.color = '#667eea';

            const rawContent = document.createElement('pre');
            rawContent.style.background = '#f8f9fa';
            rawContent.style.padding = '15px';
            rawContent.style.borderRadius = '8px';
            rawContent.style.fontSize = '12px';
            rawContent.style.overflow = 'auto';
            rawContent.style.maxHeight = '200px';
            rawContent.textContent = JSON.stringify(this.data.raw_node_data, null, 2);

            rawSection.appendChild(summary);
            rawSection.appendChild(rawContent);
            modalContent.appendChild(rawSection);
        }

        // Assemble modal
        modalContent.appendChild(closeButton);
        modalContent.appendChild(title);
        modalContent.appendChild(infoGrid);

        modal.appendChild(modalContent);
        document.body.appendChild(modal);

        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });

        // Close modal with Escape key
        document.addEventListener('keydown', function escapeHandler(e) {
            if (e.key === 'Escape') {
                modal.remove();
                document.removeEventListener('keydown', escapeHandler);
            }
        });
    }
}