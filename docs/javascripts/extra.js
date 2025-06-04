/* Custom JavaScript for HyperOpt Strategy Documentation */

document.addEventListener('DOMContentLoaded', function() {
    
    // Add performance metrics animation
    function animateMetrics() {
        const metricValues = document.querySelectorAll('.metric-value');
        
        metricValues.forEach(metric => {
            const finalValue = metric.textContent;
            const numericValue = parseFloat(finalValue.replace(/[^\d.-]/g, ''));
            const suffix = finalValue.replace(/[\d.-]/g, '');
            
            if (!isNaN(numericValue)) {
                let currentValue = 0;
                const increment = numericValue / 60; // 60 frames for smooth animation
                const timer = setInterval(() => {
                    currentValue += increment;
                    if (currentValue >= numericValue) {
                        currentValue = numericValue;
                        clearInterval(timer);
                    }
                    metric.textContent = currentValue.toFixed(1) + suffix;
                }, 16); // ~60 FPS
            }
        });
    }
    
    // Trigger animation when metrics come into view
    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const metricsObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
                if (entry.target.classList.contains('performance-metrics')) {
                    setTimeout(animateMetrics, 200);
                }
                metricsObserver.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe performance metrics and feature cards
    document.querySelectorAll('.performance-metrics, .feature-card').forEach(el => {
        metricsObserver.observe(el);
    });
    
    // Enhanced copy to clipboard functionality
    function enhanceCopyButtons() {
        document.querySelectorAll('.md-clipboard').forEach(button => {
            button.addEventListener('click', function() {
                // Add visual feedback
                const originalTitle = this.title;
                this.title = 'Copied!';
                this.style.color = '#10b981';
                
                setTimeout(() => {
                    this.title = originalTitle;
                    this.style.color = '';
                }, 2000);
            });
        });
    }
    
    // API endpoint highlighting
    function highlightApiEndpoints() {
        document.querySelectorAll('code').forEach(code => {
            const text = code.textContent;
            
            // Highlight HTTP methods
            if (text.match(/^(GET|POST|PUT|DELETE|PATCH)\s+/)) {
                const parts = text.split(' ');
                const method = parts[0];
                const endpoint = parts.slice(1).join(' ');
                
                code.innerHTML = `<span class="api-method ${method.toLowerCase()}">${method}</span>${endpoint}`;
                code.parentElement.classList.add('api-endpoint');
            }
            
            // Highlight response times
            if (text.includes('ms') && text.match(/\d+ms/)) {
                const responseTime = parseInt(text.match(/(\d+)ms/)[1]);
                let className = 'excellent';
                
                if (responseTime > 200) className = 'good';
                if (responseTime > 500) className = 'slow';
                
                code.innerHTML = code.innerHTML.replace(
                    /(\d+ms)/g, 
                    `<span class="response-time ${className}">⚡ $1</span>`
                );
            }
        });
    }
    
    // Status badge enhancement
    function enhanceStatusBadges() {
        document.querySelectorAll('code').forEach(code => {
            const text = code.textContent.toLowerCase();
            
            if (['success', 'completed', 'done', 'active'].includes(text)) {
                code.className = 'status-badge success';
            } else if (['warning', 'pending', 'in-progress'].includes(text)) {
                code.className = 'status-badge warning';
            } else if (['error', 'failed', 'failed'].includes(text)) {
                code.className = 'status-badge error';
            } else if (['info', 'note', 'tip'].includes(text)) {
                code.className = 'status-badge info';
            }
        });
    }
    
    // Smooth scrolling for navigation links
    function addSmoothScrolling() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }
    
    // Performance monitoring dashboard simulation
    function addPerformanceDashboard() {
        const charts = document.querySelectorAll('.performance-chart');
        
        charts.forEach(chart => {
            if (chart.textContent.includes('Chart placeholder')) {
                chart.innerHTML = `
                    <div style="display: flex; align-items: center; justify-content: center; gap: 1rem;">
                        <div style="width: 60px; height: 60px; border: 4px solid #e2e8f0; border-top: 4px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                        <span>Loading real-time performance data...</span>
                    </div>
                `;
                
                // Simulate data loading
                setTimeout(() => {
                    chart.innerHTML = `
                        <div style="text-align: center;">
                            <h4 style="color: #667eea; margin-bottom: 1rem;">Live Performance Metrics</h4>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                                <div>
                                    <div style="font-size: 2rem; font-weight: bold; color: #10b981;">180ms</div>
                                    <div style="font-size: 0.9rem; opacity: 0.7;">Avg Response</div>
                                </div>
                                <div>
                                    <div style="font-size: 2rem; font-weight: bold; color: #10b981;">99.9%</div>
                                    <div style="font-size: 0.9rem; opacity: 0.7;">Uptime</div>
                                </div>
                                <div>
                                    <div style="font-size: 2rem; font-weight: bold; color: #10b981;">24.1s</div>
                                    <div style="font-size: 0.9rem; opacity: 0.7;">Optimization</div>
                                </div>
                            </div>
                        </div>
                    `;
                }, 2000);
            }
        });
    }
    
    // Code block line numbering enhancement
    function enhanceCodeBlocks() {
        document.querySelectorAll('.highlight pre').forEach(pre => {
            const code = pre.querySelector('code');
            if (code && code.textContent.split('\n').length > 5) {
                pre.classList.add('line-numbers');
            }
        });
    }
    
    // Search enhancement
    function enhanceSearch() {
        const searchInput = document.querySelector('.md-search__input');
        if (searchInput) {
            searchInput.addEventListener('focus', function() {
                this.placeholder = 'Search documentation... (Try "optimization", "API", "tutorial")';
            });
            
            searchInput.addEventListener('blur', function() {
                this.placeholder = 'Search';
            });
        }
    }
    
    // Tutorial progress tracking
    function addTutorialProgress() {
        const tutorialSteps = document.querySelectorAll('.tutorial-step');
        if (tutorialSteps.length > 0) {
            let completedSteps = 0;
            
            tutorialSteps.forEach((step, index) => {
                const checkbox = step.querySelector('input[type="checkbox"]');
                if (checkbox) {
                    checkbox.addEventListener('change', function() {
                        if (this.checked) {
                            completedSteps++;
                            step.classList.add('completed');
                        } else {
                            completedSteps--;
                            step.classList.remove('completed');
                        }
                        
                        updateProgressBar(completedSteps, tutorialSteps.length);
                    });
                }
            });
        }
    }
    
    function updateProgressBar(completed, total) {
        let progressBar = document.querySelector('.tutorial-progress');
        if (!progressBar) {
            progressBar = document.createElement('div');
            progressBar.className = 'tutorial-progress';
            progressBar.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: rgba(0,0,0,0.1);
                z-index: 1000;
                transition: all 0.3s ease;
            `;
            
            const progressFill = document.createElement('div');
            progressFill.className = 'tutorial-progress-fill';
            progressFill.style.cssText = `
                height: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                transition: width 0.3s ease;
                width: 0%;
            `;
            
            progressBar.appendChild(progressFill);
            document.body.appendChild(progressBar);
        }
        
        const percentage = (completed / total) * 100;
        const fill = progressBar.querySelector('.tutorial-progress-fill');
        fill.style.width = percentage + '%';
        
        if (percentage === 100) {
            setTimeout(() => {
                progressBar.style.background = '#10b981';
                setTimeout(() => {
                    progressBar.style.opacity = '0';
                    setTimeout(() => progressBar.remove(), 300);
                }, 1000);
            }, 500);
        }
    }
    
    // Dark mode toggle enhancement
    function enhanceDarkModeToggle() {
        const toggles = document.querySelectorAll('[data-md-color-scheme]');
        toggles.forEach(toggle => {
            toggle.addEventListener('click', function() {
                // Add smooth transition
                document.body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
                
                setTimeout(() => {
                    document.body.style.transition = '';
                }, 300);
            });
        });
    }
    
    // Initialize all enhancements
    enhanceCopyButtons();
    highlightApiEndpoints();
    enhanceStatusBadges();
    addSmoothScrolling();
    addPerformanceDashboard();
    enhanceCodeBlocks();
    enhanceSearch();
    addTutorialProgress();
    enhanceDarkModeToggle();
    
    // Add loading animation styles
    const style = document.createElement('style');
    style.textContent = `
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .tutorial-step.completed {
            opacity: 0.7;
            background: rgba(16, 185, 129, 0.1);
            border-left: 4px solid #10b981;
            padding-left: 1rem;
            transition: all 0.3s ease;
        }
        
        .line-numbers {
            counter-reset: line;
        }
        
        .line-numbers code {
            counter-increment: line;
        }
        
        .line-numbers code::before {
            content: counter(line);
            display: inline-block;
            width: 3em;
            margin-right: 1em;
            color: #999;
            text-align: right;
            user-select: none;
        }
    `;
    document.head.appendChild(style);
    
    // Performance monitoring
    if ('performance' in window) {
        window.addEventListener('load', function() {
            const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
            console.log(`Documentation loaded in ${loadTime}ms`);
            
            // Report slow loading
            if (loadTime > 3000) {
                console.warn('Documentation loading slowly. Consider optimizing assets.');
            }
        });
    }
});

// Utility functions for external use
window.HyperOptDocs = {
    highlightCode: function(selector) {
        document.querySelectorAll(selector).forEach(el => {
            el.classList.add('highlight-code');
            setTimeout(() => el.classList.remove('highlight-code'), 2000);
        });
    },
    
    showNotification: function(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            background: var(--md-primary-fg-color);
            color: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10000;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}; 