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

    
    async function initialize() {
        try {
            pyodide = await loadPyodide();
            await pyodide.loadPackage(['numpy', 'scipy']);
            
            pythonCode = await (await fetch('./code.py')).text();

           
            loaderOverlay.style.display = 'none';
            calculateBtn.disabled = false;
            calculateBtn.textContent = 'Plot';
            
            
            calculateAndPlot();
        } catch (error) {
            console.error('Pyodide initialization failed:', error);
            loaderOverlay.innerHTML = `<div class=\"loader-content\"><p>Error initializing environment.</p><small>${error.message}</small></div>`;
        }
    }

    
    async function calculateAndPlot() {
        if (!pyodide) {
            statusMessage.textContent = 'Error: Python environment not loaded.';
            statusMessage.className = 'status error';
            return;
        }

        const expr = expressionInput.value;
        const minSample = minSampleInput.value;

        
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

           
            statusMessage.textContent = ''; 
            statusMessage.className = 'status';

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
                  
                    data: plotData.y,
                    borderColor: 'rgb(0, 191, 255)', 
                    backgroundColor: 'rgba(0, 191, 255, 0.2)', 
                    fill: true,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0 
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: true,
                scales: {
                    x: { type: 'linear', title: { display: true, text: 'Value' } },
                    y: { beginAtZero: true, title: { display: true, true: 'Probability Density' } }
                },
                plugins: {
                    title: { display: false },
                    legend: { 
                        display: false
                    }
                }
            }
        });
    };

    calculateBtn.addEventListener('click', calculateAndPlot);
    initialize();
});
