// Function to fetch analytics data and update stats
async function fetchAnalyticsData() {
    try {
        const response = await fetch('/analytics/data');
        if (response.ok) {
            const data = await response.json();
            
            // Update statistics
            document.getElementById('total-objects').textContent = data.total_objects || '--';
            document.getElementById('total-evaluations').textContent = data.total_evaluations || '--';
            document.getElementById('unknown-objects').textContent = data.unknown_objects || '--';
            document.getElementById('hallucinations').textContent = data.hallucinations || '--';
        }
    } catch (error) {
        console.error('Error fetching analytics data:', error);
    }
}

// Function to refresh all images
function refreshImages() {
    const timestamp = new Date().getTime();
    document.getElementById('ratings-img').src = '/analytics/histogram/ratings?' + timestamp;
    document.getElementById('time-img').src = '/analytics/distribution/time_spent?' + timestamp;
    document.getElementById('unknown-img').src = '/analytics/unknown_count?' + timestamp;
    document.getElementById('hallucination-img').src = '/analytics/hallucination_count?' + timestamp;
    document.getElementById('disagreement-img').src = '/analytics/rating_disagreement?' + timestamp;
    
    // Also refresh the data
    fetchAnalyticsData();
}

// Initialize data on page load
document.addEventListener('DOMContentLoaded', function() {
    fetchAnalyticsData();
});

// Refresh every 5 minutes
setInterval(refreshImages, 5 * 60 * 1000);