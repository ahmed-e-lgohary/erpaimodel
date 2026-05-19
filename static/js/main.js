document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const emptyState = document.getElementById('emptyState');
    const loading = document.getElementById('loading');
    const resultDisplay = document.getElementById('result');
    const errorState = document.getElementById('errorState');
    
    const passengerCount = document.getElementById('passengerCount');
    const resultDetails = document.getElementById('resultDetails');
    const recommendationText = document.getElementById('recommendationText');
    const errorMessage = document.getElementById('errorMessage');

    // Set today's date and current time as default
    const now = new Date();
    document.getElementById('date').value = now.toISOString().split('T')[0];
    document.getElementById('time').value = now.toTimeString().slice(0, 5);

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Hide other states, show loading
        emptyState.classList.add('hidden');
        resultDisplay.classList.add('hidden');
        errorState.classList.add('hidden');
        loading.classList.remove('hidden');

        const formData = new FormData(form);
        const data = {
            date: formData.get('date'),
            time: formData.get('time'),
            area: formData.get('area')
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            const result = await response.json();

            // Hide loading
            loading.classList.add('hidden');

            if (response.ok) {
                // Show success result
                animateValue(passengerCount, 0, result.passengers, 1000);
                resultDetails.textContent = `Expected in ${result.area} on ${data.date} at ${data.time}`;
                
                // Simple logic for recommendation
                let rec = "";
                if (result.passengers > 100) {
                    rec = "High demand expected! Deploy 3-4 extra vehicles to this zone to cover the rush.";
                } else if (result.passengers > 50) {
                    rec = "Moderate demand. Standard vehicle deployment should be sufficient.";
                } else {
                    rec = "Low demand expected. Consider reducing active vehicles in this zone to save costs.";
                }
                recommendationText.textContent = rec;
                
                resultDisplay.classList.remove('hidden');
            } else {
                // Show error
                errorMessage.textContent = result.error || "An unknown error occurred.";
                errorState.classList.remove('hidden');
            }
        } catch (error) {
            loading.classList.add('hidden');
            errorMessage.textContent = "Failed to connect to the server.";
            errorState.classList.remove('hidden');
        }
    });

    // Animate number counting up
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            // easeOutQuart
            const easeProgress = 1 - Math.pow(1 - progress, 4);
            obj.innerHTML = Math.floor(easeProgress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.innerHTML = end; // Ensure final value is exact
            }
        };
        window.requestAnimationFrame(step);
    }

    // --- Dashboard Logic ---
    let hourlyChartInstance = null;
    let distributionChartInstance = null;
    let dashboardData = null;

    async function initDashboard() {
        try {
            const response = await fetch('/api/dashboard_data');
            if (response.ok) {
                dashboardData = await response.json();
                renderDistributionChart(dashboardData.total_per_area);
                
                const areaSelect = document.getElementById('area');
                const defaultArea = areaSelect.value || Object.keys(dashboardData.hourly_trend)[0];
                renderHourlyChart(dashboardData.hourly_trend, defaultArea);
            }
        } catch (error) {
            console.error("Failed to load dashboard data", error);
        }
    }

    document.getElementById('area').addEventListener('change', (e) => {
        if (dashboardData && dashboardData.hourly_trend) {
            renderHourlyChart(dashboardData.hourly_trend, e.target.value);
        }
    });

    function renderHourlyChart(hourlyTrendData, selectedArea) {
        const ctx = document.getElementById('hourlyTrendChart').getContext('2d');
        if (hourlyChartInstance) hourlyChartInstance.destroy();

        const dataPoints = hourlyTrendData[selectedArea] || Array(24).fill(0);
        const labels = Array.from({length: 24}, (_, i) => `${i}:00`);

        let gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, 'rgba(79, 70, 229, 0.5)'); // --primary glow
        gradient.addColorStop(1, 'rgba(79, 70, 229, 0.0)');

        hourlyChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: `Avg Demand - ${selectedArea}`,
                    data: dataPoints,
                    borderColor: '#6366f1',
                    backgroundColor: gradient,
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#ec4899',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#ec4899'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#e2e8f0' } }
                },
                scales: {
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });
    }

    function renderDistributionChart(totalPerAreaData) {
        const ctx = document.getElementById('areaDistributionChart').getContext('2d');
        if (distributionChartInstance) distributionChartInstance.destroy();

        distributionChartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(totalPerAreaData),
                datasets: [{
                    data: Object.values(totalPerAreaData),
                    backgroundColor: ['#4f46e5', '#ec4899', '#06b6d4', '#f59e0b', '#10b981', '#8b5cf6'],
                    borderWidth: 2,
                    borderColor: '#0f172a',
                    hoverOffset: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: { color: '#e2e8f0', padding: 15 }
                    }
                },
                cutout: '70%'
            }
        });
    }

    initDashboard();
});
