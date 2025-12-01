/* Student Engagement System - JavaScript */

class EngagementDashboard {
    constructor() {
        this.socket = null;
        this.chart = null;
        this.history = [];
    }
    
    init() {
        console.log('Student Engagement Dashboard initialized');
        this.setupChart();
        this.updateStats({
            totalStudents: 0,
            averageScore: 0,
            engagementLevel: 'neutral'
        });
    }
    
    setupChart() {
        // Setup engagement chart
        console.log('Chart setup complete');
    }
    
    updateStats(stats) {
        const statsElement = document.getElementById('stats');
        if (statsElement) {
            statsElement.innerHTML = `
                <div class="stat-item">
                    <span class="stat-label">Số sinh viên:</span>
                    <span class="stat-value">${stats.totalStudents}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Điểm TB:</span>
                    <span class="stat-value">${stats.averageScore.toFixed(1)}%</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Mức độ:</span>
                    <span class="stat-value ${stats.engagementLevel}">${this.getLevelLabel(stats.engagementLevel)}</span>
                </div>
            `;
        }
    }
    
    getLevelLabel(level) {
        const labels = {
            'highly_engaged': 'Rất hứng thú',
            'engaged': 'Hứng thú',
            'neutral': 'Bình thường',
            'disengaged': 'Không hứng thú',
            'highly_disengaged': 'Rất không hứng thú'
        };
        return labels[level] || level;
    }
    
    addDataPoint(score) {
        this.history.push(score);
        if (this.history.length > 50) {
            this.history.shift();
        }
        this.updateChart();
    }
    
    updateChart() {
        // Update chart with new data
        console.log('Chart updated with', this.history.length, 'points');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    const dashboard = new EngagementDashboard();
    dashboard.init();
});
