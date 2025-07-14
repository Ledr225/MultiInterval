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
        statusMessage.className = 'status info';
        calculateBtn.disabled = true;

        try {
            // Ensure the Python code is re-run with the latest version
            await pyodide.runPythonAsync(pythonCode);
            const pythonResultPyProxy = await pyodide.globals.get('run_calculation')(expr, minSample);
            
            // --- MODIFICATION HERE: Use pyodide.toJs() with `dict_converter` for full conversion ---
            // Convert PyProxy to JS object, ensuring all nested dicts are converted
            const data = pythonResultPyProxy.toJs({ dict_converter: Object.fromEntries });
            pythonResultPyProxy.destroy(); // Clean up PyProxy object

            // --- IMPORTANT: Robust checks for plotData ---
            if (!data) {
                statusMessage.textContent = 'Error: Python calculation returned no data or an invalid data format.';
                statusMessage.className = 'status error';
                console.error('Python calculation returned null or undefined data:', pythonResultPyProxy); // Log the original PyProxy if needed
                return;
            }

            if (data.error) {
                statusMessage.textContent = `Error: ${data.error}`;
                statusMessage.className = 'status error';
                console.error(data.error);
                return;
            }

            // After the recursive toJs conversion, plot_data should now reliably be a plain object
            const plotDataForChart = data.plot_data;

            // Final check that x and y are arrays
            if (!plotDataForChart || !Array.isArray(plotDataForChart.x) || !Array.isArray(plotDataForChart.y)) {
                statusMessage.textContent = 'Error: Plot data (x/y components) are not valid arrays. Cannot render chart.';
                statusMessage.className = 'status error';
                console.error('Plot data components are not arrays after conversion:', plotDataForChart);
                return;
            }
            // --- End of robust checks ---

            statusMessage.textContent = data.status || 'Calculation successful.';
            statusMessage.className = 'status success';
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
                labels: plotData.x, // These are now guaranteed to be arrays by the checks above
                datasets: [{
                    label: 'Approximate Probability Density',
                    data: plotData.y, // These are now guaranteed to be arrays by the checks above
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    fill: true,
                    borderWidth: 2,
                    pointRadius: 0,
                    // Set tension to 0 for straight lines between points, removing Chart.js Bezier smoothing
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
                    title: { display: true, text: `Result Distribution for: ${expressionInput.value}` }
                }
            }
        });
    };

    calculateBtn.addEventListener('click', calculateAndPlot);
    initialize();
});
