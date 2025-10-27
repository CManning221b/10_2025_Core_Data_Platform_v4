// static/js/visualisation/core_plot.js
class CorePlot {
    constructor(x, parentID, exclusions, dataArray, options = {}) {
        this.x = x;
        this.container = document.getElementById(parentID);
        this.grid = [];
        this.exclusions = exclusions;
        this.values = dataArray;
        this.reactorType = options.reactorType || 'CANDU';

        // Calculate responsive sizing
        const containerWidth = this.container.clientWidth || window.innerWidth * 0.8;
        const containerHeight = window.innerHeight * 0.7;
        this.maxTableWidth = Math.min(containerWidth, containerHeight);

        console.log('CorePlot initialized with data:', dataArray);
        console.log('Reactor type:', this.reactorType);
    }

    createGrid() {
        console.log('Creating grid for reactor type:', this.reactorType);

        if (this.reactorType === 'AGR') {
            this.createAGRGrid();
        } else {
            this.createCANDUGrid();
        }
    }
    getDisplayName(realChannelName) {
        return this.channelDisplayMapping[realChannelName] || realChannelName;
    }

    createCANDUGrid() {
        this.container.innerHTML = '';

        const gridContainer = document.createElement('div');
        gridContainer.className = 'core-plot-grid';

        const table = document.createElement('table');
        table.className = 'channel-grid-table';

        // ADDED: Center the table for circular pattern
        table.style.margin = '0 auto';
        table.style.borderCollapse = 'separate';
        table.style.borderSpacing = '2px';

        // Create rows and columns (your original logic)
        for (let i = 0; i < this.x; i++) {
            const row = document.createElement('tr');
            row.style.textAlign = 'center';
            const rowArray = [];

            for (let j = 0; j < this.x; j++) {
                const cell = document.createElement('td');
                cell.className = 'channel-cell';

                let rowLabel = String.fromCharCode(65 + i);
                if (i >= 8) {
                    rowLabel = String.fromCharCode(66 + i);
                }

                const colNumber = (j + 1 < 10) ? `0${j + 1}` : j + 1;
                const realChannelName = `${rowLabel}${colNumber}`;

                // ANONYMIZE: A01 → 0101
                const letterPos = realChannelName.charCodeAt(0) - 65 + 1;
                const anonChannelName = String(letterPos).padStart(2, '0') + realChannelName.slice(1);

                cell.id = anonChannelName;
                cell.dataset.channel = anonChannelName;

                if (this.exclusions.includes(anonChannelName)) {
                    cell.style.visibility = 'hidden';
                    cell.style.width = '20px';
                    cell.style.height = '20px';
                } else {
                    cell.style.visibility = 'visible';
                }

                row.appendChild(cell);
                rowArray.push(cell);
            }

            table.appendChild(row);
            this.grid.push(rowArray);
        }

        gridContainer.appendChild(table);
        this.container.appendChild(gridContainer);

        this.populateGrid(this.values);
    }


createAGRGrid() {
    // Create a reactor core pattern for the 8 AGR channels
    this.container.innerHTML = '';

    const gridContainer = document.createElement('div');
    gridContainer.className = 'core-plot-grid';

    const table = document.createElement('table');
    table.className = 'channel-grid-table agr-table';
    table.style.margin = '0 auto';
    table.style.borderCollapse = 'separate';
    table.style.borderSpacing = '4px';
    table.style.backgroundColor = '#ffffff'; // CHANGED: White background like CANDU
    table.style.padding = '15px';
    table.style.borderRadius = '15px';
    table.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)'; // CHANGED: Lighter shadow

    // Your 8 AGR channels (actual data)
    const agrChannels = ['0878', '1270', '1288', '1464', '1858', '1890', '2094', '2274'];

    // Valid AGR placeholder channels (fit the pattern but no data)
    const validAGRPlaceholders = [
        '1234', '1456', '1678', '2468', '2680', '3024', '3246', '3468',
        '3680', '4024', '4246', '4468', '4680', '5024', '5246', '5468',
        '5680', '6024', '6246', '6468', '6680', '7024', '7246', '7468',
        '8024', '8246', '8468', '8680', '9024', '9246', '9468'
    ];

    // Reactor core pattern layout (7x7 grid for realistic look)
    const corePattern = [
        [null, null, null, 0, null, null, null],
        [null, null, 1, null, 2, null, null],
        [null, 3, null, 4, null, 5, null],
        [6, null, null, 7, null, null, null],
        [null, null, null, null, null, null, null],
        [null, null, null, null, null, null, null],
        [null, null, null, null, null, null, null]
    ];

    let placeholderIndex = 0;

    // Create the core grid
    for (let i = 0; i < 7; i++) {
        const row = document.createElement('tr');
        row.style.textAlign = 'center';
        const rowArray = [];

        for (let j = 0; j < 7; j++) {
            const cell = document.createElement('td');
            cell.className = 'channel-cell';

            const channelIndex = corePattern[i][j];

            if (channelIndex !== null && channelIndex < agrChannels.length) {
                // Real channel position with actual data
                const channelName = agrChannels[channelIndex];
                cell.id = channelName;
                cell.dataset.channel = channelName;
                cell.dataset.hasData = 'true';
            } else {
                // Empty position - fill with valid AGR placeholder
                if (placeholderIndex < validAGRPlaceholders.length) {
                    const placeholderName = validAGRPlaceholders[placeholderIndex];
                    cell.id = placeholderName;
                    cell.dataset.channel = placeholderName;
                    cell.dataset.isPlaceholder = 'true';
                    placeholderIndex++;
                } else {
                    // Truly empty if we run out of placeholders
                    cell.style.visibility = 'hidden';
                }
            }

            row.appendChild(cell);
            rowArray.push(cell);
        }

        table.appendChild(row);
        this.grid.push(rowArray);
    }

    gridContainer.appendChild(table);
    this.container.appendChild(gridContainer);

    this.populateGrid(this.values);
}

populateGrid(cellData) {
        console.log('Populating grid with channel data:', cellData);
        this.clearGrid();
        this.values = cellData;

        // Bigger cells for AGR since fewer channels
        const cellSize = this.reactorType === 'AGR' ?
            Math.min(80, this.maxTableWidth / 3) :
            this.maxTableWidth / this.x;

        this.grid.forEach((row, i) => {
            row.forEach((cell, j) => {
                const channelName = cell.id;

                // Skip hidden/excluded cells - they're already invisible placeholders
                if (!channelName || cell.style.visibility === 'hidden') {
                    return;
                }

                const channelData = cellData[channelName] || null;
                const color = this.getChannelColor(channelData);

                console.log(`Channel ${channelName}:`, { data: channelData, color });

                const buttonData = {
                    name: channelName,
                    value: channelData ? channelData.measurement_count : 0,
                    status: channelData ? channelData.status : 'not_found',
                    last_measurement: channelData ? channelData.last_measurement : null,
                    ...channelData
                };

                this.createButton(channelName, cellSize, buttonData, cell, color);
            });
        });
    }

    getChannelColor(channelData) {
        if (!channelData) {
            return 'rgba(204,204,204,0.6)';  // gray - not in graph
        } else if (channelData.measurement_count === 0) {
            return '#FFC600';  // yellow - in graph but no measurements
        } else if (channelData.measurement_count >= 10) {
            return '#0066CC';  // blue - rich temporal data
        } else if (channelData.measurement_count >= 3) {
            return '#00AA00';  // darker green - good measurements
        } else {
            return '#66DD66';  // light green - some measurements
        }
    }

    clearGrid() {
        this.grid.forEach((row) => {
            row.forEach((cell) => {
                while (cell.firstChild) {
                    cell.removeChild(cell.firstChild);
                }
            });
        });
    }


    createButton(channelName, size, val, cell, color) {
        const buttonData = {
            name: channelName,  // Already anonymized
            value: val.measurement_count || 0,
            status: val.status || 'unknown',
            last_measurement: val.last_measurement,
            ...val
        };

        const channel = new ChannelButton(buttonData, color, 'name', 'value', size);
        const button = channel.createButton();
        cell.appendChild(button);
    }
}