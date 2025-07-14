document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const calculateBtn = document.getElementById('calculate-btn');
    const expressionInput = document.getElementById('expression');
    const minSampleInput = document.getElementById('min_sample');
    const statusMessage = document.getElementById('status-message');
    const loaderOverlay = document.getElementById('loader-overlay');
    const ctx = document.getElementById('result-chart').getContext('2d');
    let resultChart = null;
    let pyodide = null;
    let pythonCode = '';

    // --- Main Initialization ---
    async function initialize() {
        try {
            pyodide = await loadPyodide();
            await pyodide.loadPackage(['numpy', 'scipy']);
            // Fetch the Python code from code.py
            pythonCode = await (await fetch('./code.py')).text();

            // Hide loader and enable button
            loaderOverlay.style.display = 'none';
            calculateBtn.disabled = false;
            calculateBtn.textContent = 'Calculate & Plot';
            
            // Run a default calculation on load
            calculateAndPlot();
        } catch (error) {
            console.error('Pyodide initialization failed:', error);
            loaderOverlay.innerHTML = `<div class=\"loader-content\"><p>Error initializing environment.</p><small>${error.message}</small></div>`;
        }
    }

    // --- Calculation Logic ---
    async function calculateAndPlot() {
        if (!pyodide) {
            statusMessage.textContent = 'Error: Python environment not loaded.';
            statusMessage.className = 'status error';
            return;
        }

        const expr = expressionInput.value;
        const minSample = minSampleInput.value;

        statusMessage.textContent = 'Calculating...';
        statusMessage.className = 'status loading'; // Set to loading state
        calculateBtn.disabled = true;

        try {
            await pyodide.runPythonAsync(pythonCode);
            const pythonResultPyProxy = await pyodide.globals.get('run_calculation')(expr, minSample);
            
            const data = pythonResultPyProxy.toJs({ dict_converter: Object.fromEntries });
            pythonResultPyProxy.destroy();

            if (data.error) {
                statusMessage.textContent = `Error: ${data.error}`;
                statusMessage.className = 'status error';
                console.error(data.error);
                return;
            }

            let plotDataForChart = null;
            if (data.plot_data instanceof Map) {
                plotDataForChart = {
                    x: data.plot_data.get('x'),
                    y: data.plot_data.get('y')
                };
            } else if (data.plot_data && typeof data.plot_data === 'object') {
                plotDataForChart = data.plot_data;
            } else {
                statusMessage.textContent = 'Error: Plot data is missing or has an unexpected format.';
                statusMessage.className = 'status error';
                console.error('Received data object has invalid plot_data:', data);
                return;
            }

            if (!plotDataForChart || !Array.isArray(plotDataForChart.x) || !Array.isArray(plotDataForChart.y)) {
                statusMessage.textContent = 'Error: Plot data (x/y components) are not valid arrays. Cannot render chart.';
                statusMessage.className = 'status error';
                console.error('Plot data components are not arrays:', plotDataForChart);
                return;
            }

            // Always hide status message on successful plot rendering
            statusMessage.textContent = ''; // Clear text content
            statusMessage.className = 'status'; // Reset class to `display: none;` from CSS

            renderChart(plotDataForChart);

        } catch (error) {
            statusMessage.textContent = `Error: ${error.message}`;
            statusMessage.className = 'status error';
            console.error(error);
        } finally {
            calculateBtn.disabled = false;
        }
    }

    // --- Chart Rendering ---
    const renderChart = (plotData) => {
        if (resultChart) resultChart.destroy();
        resultChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: plotData.x, 
                datasets: [{
                    label: 'Approximate Probability Density',
                    data: plotData.y,
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    fill: true,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0 
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { type: 'linear', title: { display: true, text: 'Value' } },
                    y: { beginAtZero: true, title: { display: true, text: 'Probability Density' } }
                },
                plugins: {
                    // Removed the chart title
                    title: { display: false } 
                }
            }
        });
    };

    calculateBtn.addEventListener('click', calculateAndPlot);
    initialize();
});
