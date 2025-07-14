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
            loaderOverlay.innerHTML = `<div class="loader-content"><p>Error initializing environment.</p><small>${error.message}</small></div>`;
        }
    }

    // --- Calculation Logic ---
    async function calculateAndPlot() {
        if (!pyodide) {
            statusMessage.textContent = 'Pyodide is not ready.';
            statusMessage.className = 'status error';
            return;
        }

        // Show loading state
        statusMessage.textContent = 'Calculating...';
        statusMessage.className = 'status loading';
        if (resultChart) {
            resultChart.destroy();
        }

        const expression = expressionInput.value;
        const min_sample = minSampleInput.value;

        try {
            // Run the Python code in the Pyodide environment
            pyodide.runPython(pythonCode);
            // Get the main function from Python
            const runCalculation = pyodide.globals.get('run_calculation');
            // Call the function and get the result
            const result = runCalculation(expression, min_sample);
            const data = result.toJs({ dict_converter: Object.fromEntries }); // Convert Python dict to JS object
            result.destroy(); // Clean up memory

            if (data.error) {
                throw new Error(data.error);
            }
            
            statusMessage.textContent = data.status;
            statusMessage.className = 'status success';
            renderChart(data.plot_data);

        } catch (error) {
            statusMessage.textContent = `Error: ${error.message}`;
            statusMessage.className = 'status error';
            console.error(error);
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
                    tension: 0.1
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