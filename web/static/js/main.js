// Common JavaScript functionality

// Flash message auto-dismiss
document.addEventListener('DOMContentLoaded', function() {
    // Auto-dismiss flash messages after 5 seconds
    const flashMessages = document.querySelectorAll('.alert');
    flashMessages.forEach(function(message) {
        setTimeout(function() {
            message.style.opacity = '0';
            setTimeout(function() {
                message.remove();
            }, 300);
        }, 5000);
    });

    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });

    // Confirm delete actions
    const deleteButtons = document.querySelectorAll('.delete-confirm');
    deleteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            if (!confirm('Are you sure you want to delete this item?')) {
                e.preventDefault();
            }
        });
    });

    // Refresh data periodically (for dashboard)
    if (document.querySelector('.auto-refresh')) {
        setInterval(function() {
            const refreshUrl = document.querySelector('.auto-refresh').dataset.refreshUrl;
            if (refreshUrl) {
                fetch(refreshUrl)
                    .then(response => response.json())
                    .then(data => {
                        // Update dashboard elements
                        Object.keys(data).forEach(key => {
                            const element = document.querySelector(`[data-update="${key}"]`);
                            if (element) {
                                element.textContent = data[key];
                            }
                        });
                    })
                    .catch(error => console.error('Error refreshing data:', error));
            }
        }, 30000); // Refresh every 30 seconds
    }
});

// API request helper
async function apiRequest(url, method = 'GET', data = null) {
    try {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': document.querySelector('meta[name="api-key"]')?.content
            }
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(url, options);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// Chart initialization helper
function initChart(canvasId, type, data, options = {}) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type,
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            ...options
        }
    });
}

// File upload helper
function handleFileUpload(input, previewId) {
    const file = input.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById(previewId);
            if (preview) {
                preview.src = e.target.result;
            }
        };
        reader.readAsDataURL(file);
    }
}

// Export functions for use in other files
window.apiRequest = apiRequest;
window.initChart = initChart;
window.handleFileUpload = handleFileUpload; 