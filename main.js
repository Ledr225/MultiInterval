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
            const pythonResult = await pyodide.globals.get('run_calculation')(expr, minSample);
            
            // Convert PyProxy to JS object if necessary
            const data = pythonResult.toJs ? pythonResult.toJs({ dictConverter: Object.fromEntries }) : pythonResult;
            pythonResult.destroy(); // Clean up PyProxy object

            if (data.error) {
                statusMessage.textContent = `Error: ${data.error}`;
                statusMessage.className = 'status error';
                console.error(data.error);
                return;
            }

            statusMessage.textContent = data.status || 'Calculation successful.';
            statusMessage.className = 'status success';
            renderChart(data.plot_data);

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
                    tension: 0.1 // Default Chart.js tension
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
