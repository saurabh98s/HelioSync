// Dashboard-specific JavaScript functionality

document.addEventListener('DOMContentLoaded', function() {
    // Initialize dashboard charts
    initializeCharts();
    
    // Initialize real-time updates
    initializeRealTimeUpdates();
    
    // Initialize project status indicators
    initializeProjectStatus();
});

function initializeCharts() {
    // Training Progress Chart
    const trainingProgressCtx = document.getElementById('trainingProgressChart');
    if (trainingProgressCtx) {
        const trainingProgressChart = new Chart(trainingProgressCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#007bff',
                    tension: 0.1
                }, {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: '#28a745',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // Client Participation Chart
    const clientParticipationCtx = document.getElementById('clientParticipationChart');
    if (clientParticipationCtx) {
        const clientParticipationChart = new Chart(clientParticipationCtx, {
            type: 'doughnut',
            data: {
                labels: ['Active', 'Inactive', 'Training'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#28a745', '#dc3545', '#ffc107']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
}

function initializeRealTimeUpdates() {
    // Update dashboard stats every 30 seconds
    setInterval(updateDashboardStats, 30000);
    
    // Update charts every minute
    setInterval(updateCharts, 60000);
}

async function updateDashboardStats() {
    try {
        const response = await apiRequest('/api/dashboard/stats');
        updateStatsDisplay(response);
    } catch (error) {
        console.error('Failed to update dashboard stats:', error);
    }
}

function updateStatsDisplay(stats) {
    // Update project counts
    const projectCountElement = document.getElementById('projectCount');
    if (projectCountElement) {
        projectCountElement.textContent = stats.total_projects;
    }

    // Update client counts
    const clientCountElement = document.getElementById('clientCount');
    if (clientCountElement) {
        clientCountElement.textContent = stats.total_clients;
    }

    // Update active training sessions
    const activeTrainingElement = document.getElementById('activeTraining');
    if (activeTrainingElement) {
        activeTrainingElement.textContent = stats.active_training;
    }

    // Update model versions
    const modelVersionsElement = document.getElementById('modelVersions');
    if (modelVersionsElement) {
        modelVersionsElement.textContent = stats.total_models;
    }
}

async function updateCharts() {
    try {
        const response = await apiRequest('/api/dashboard/chart-data');
        updateChartsDisplay(response);
    } catch (error) {
        console.error('Failed to update charts:', error);
    }
}

function updateChartsDisplay(data) {
    // Update training progress chart
    const trainingProgressChart = Chart.getChart('trainingProgressChart');
    if (trainingProgressChart) {
        trainingProgressChart.data.labels = data.training_progress.labels;
        trainingProgressChart.data.datasets[0].data = data.training_progress.training_loss;
        trainingProgressChart.data.datasets[1].data = data.training_progress.validation_loss;
        trainingProgressChart.update();
    }

    // Update client participation chart
    const clientParticipationChart = Chart.getChart('clientParticipationChart');
    if (clientParticipationChart) {
        clientParticipationChart.data.datasets[0].data = [
            data.client_participation.active,
            data.client_participation.inactive,
            data.client_participation.training
        ];
        clientParticipationChart.update();
    }
}

function initializeProjectStatus() {
    // Add status indicators to project cards
    const projectCards = document.querySelectorAll('.project-card');
    projectCards.forEach(card => {
        const status = card.dataset.status;
        const statusIndicator = document.createElement('div');
        statusIndicator.className = `status-indicator status-${status}`;
        card.appendChild(statusIndicator);
    });
}

// Export functions for use in other files
window.initializeCharts = initializeCharts;
window.updateDashboardStats = updateDashboardStats;
window.updateCharts = updateCharts; 