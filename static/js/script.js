/**
 * EduPredict - Student Performance Prediction System
 * Frontend Interactive Controller & Chart.js Engine
 */

document.addEventListener('DOMContentLoaded', () => {
    initNavToggle();
    initPredictionForm();
    initHistoryControls();
    
    // Page-specific initializers based on element existence
    if (document.getElementById('chartActualVsPred')) {
        initPerformanceCharts();
    }
    if (document.getElementById('chartCategoryDoughnut')) {
        initInsightsCharts();
    }
    if (document.getElementById('homeCategoryChart')) {
        initHomePreviewChart();
    }
});

/* -------------------------------------------------------------
   NAVIGATION TOGGLE
   ------------------------------------------------------------- */
function initNavToggle() {
    const toggleBtn = document.querySelector('.nav-toggle-btn');
    const navLinks = document.querySelector('.nav-links');
    if (toggleBtn && navLinks) {
        toggleBtn.addEventListener('click', () => {
            navLinks.classList.toggle('show');
        });
    }
}

/* -------------------------------------------------------------
   PREDICTION FORM CONTROLLER
   ------------------------------------------------------------- */
function initPredictionForm() {
    const form = document.getElementById('predictionForm');
    if (!form) return;

    // Synchronize Range Sliders and Number Inputs
    const syncPairs = [
        { slider: 'study_hours_slider', num: 'study_hours_num', display: 'disp_study_hours' },
        { slider: 'attendance_slider', num: 'attendance_num', display: 'disp_attendance' },
        { slider: 'prev_score_slider', num: 'prev_score_num', display: 'disp_prev_score' },
        { slider: 'assignment_slider', num: 'assignment_num', display: 'disp_assignment' },
        { slider: 'sleep_slider', num: 'sleep_num', display: 'disp_sleep' },
        { slider: 'participation_slider', num: 'participation_num', display: 'disp_participation' },
        { slider: 'backlogs_slider', num: 'backlogs_num', display: 'disp_backlogs' }
    ];

    syncPairs.forEach(pair => {
        const slider = document.getElementById(pair.slider);
        const num = document.getElementById(pair.num);
        const disp = document.getElementById(pair.display);

        if (slider && num) {
            slider.addEventListener('input', (e) => {
                num.value = e.target.value;
                if (disp) disp.textContent = e.target.value;
            });
            num.addEventListener('input', (e) => {
                slider.value = e.target.value;
                if (disp) disp.textContent = e.target.value;
            });
        }
    });

    // Demo Data Preset Buttons
    const demoBtns = document.querySelectorAll('.demo-preset-btn');
    demoBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const presetKey = btn.dataset.preset || 'recommended';
            loadDemoPreset(presetKey);
        });
    });

    // Handle AJAX Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await handleFormSubmit(form);
    });
}

/**
 * Loads realistic demo profiles into the form
 */
async function loadDemoPreset(presetKey) {
    try {
        const res = await fetch('/api/demo-data');
        const data = await res.json();
        const profile = data[presetKey] || data['recommended'];

        if (!profile) return;

        // Set text
        const nameField = document.getElementById('student_name');
        if (nameField) nameField.value = profile.student_name;

        // Set synchronized numerical inputs
        setFieldValue('study_hours', profile.study_hours_per_day);
        setFieldValue('attendance', profile.attendance_percentage);
        setFieldValue('prev_score', profile.previous_score);
        setFieldValue('assignment', profile.assignment_completion);
        setFieldValue('sleep', profile.sleep_hours);
        setFieldValue('participation', profile.class_participation);
        setFieldValue('backlogs', profile.backlogs);

        // Set radio / segmented controls
        setRadioValue('internet_availability', profile.internet_availability);
        setRadioValue('device_availability', profile.device_availability);
        setRadioValue('extracurricular_activity', profile.extracurricular_activity);
        setRadioValue('parental_support', profile.parental_support);

        // Flash message toast
        showToast(`Loaded preset profile: ${profile.student_name}`, 'info');
    } catch (err) {
        console.error('Error loading demo preset:', err);
    }
}

function setFieldValue(prefix, val) {
    const slider = document.getElementById(`${prefix}_slider`);
    const num = document.getElementById(`${prefix}_num`);
    const disp = document.getElementById(`disp_${prefix}`);

    if (slider) slider.value = val;
    if (num) num.value = val;
    if (disp) disp.textContent = val;
}

function setRadioValue(name, val) {
    const radio = document.querySelector(`input[name="${name}"][value="${val}"]`);
    if (radio) radio.checked = true;
}

/**
 * Sends prediction request to /predict and renders results
 */
async function handleFormSubmit(form) {
    const submitBtn = document.getElementById('predictSubmitBtn');
    const originalBtnText = submitBtn ? submitBtn.innerHTML : 'Predict Performance';

    try {
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing with AI Model...';
        }

        const formData = new FormData(form);
        const payload = Object.fromEntries(formData.entries());

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (!response.ok || !result.success) {
            const errs = result.errors ? result.errors.join('<br>') : 'Prediction failed. Check input values.';
            showToast(errs, 'danger');
            return;
        }

        // Render result card
        renderPredictionResult(result);
        showToast('Prediction completed successfully!', 'success');

        // Scroll into view on mobile
        const resultSection = document.getElementById('predictionResultCard');
        if (resultSection && window.innerWidth < 992) {
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

    } catch (err) {
        console.error('Submission error:', err);
        showToast('An unexpected network error occurred.', 'danger');
    } finally {
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalBtnText;
        }
    }
}

/**
 * Updates UI with prediction metrics, gauge, and recommendations
 */
function renderPredictionResult(data) {
    const emptyState = document.getElementById('resultEmptyState');
    const resultDetails = document.getElementById('resultDetails');
    if (emptyState) emptyState.style.display = 'none';
    if (resultDetails) resultDetails.style.display = 'block';

    // Animate Score Counter
    const scoreValEl = document.getElementById('resultScoreVal');
    const scoreCircle = document.getElementById('resultScoreCircle');
    const targetScore = data.predicted_score;

    animateCounter(scoreValEl, 0, targetScore, 750);

    // Update Circle Color
    if (scoreCircle) {
        scoreCircle.style.borderColor = data.color;
        scoreCircle.style.boxShadow = `0 8px 24px ${data.color}33`;
    }

    // Category Badge
    const categoryBadge = document.getElementById('resultCategoryBadge');
    if (categoryBadge) {
        categoryBadge.className = `badge ${data.badge_class}`;
        categoryBadge.textContent = data.category;
    }

    // Category Summary
    const summaryEl = document.getElementById('resultSummary');
    if (summaryEl) {
        summaryEl.textContent = data.summary;
    }

    // Confidence Level
    const confValEl = document.getElementById('resultConfidenceVal');
    const confBarEl = document.getElementById('resultConfidenceBar');
    if (confValEl) confValEl.textContent = `${data.confidence}%`;
    if (confBarEl) confBarEl.style.width = `${data.confidence}%`;

    // Personalized Recommendations List
    const recsListEl = document.getElementById('resultRecommendationsList');
    if (recsListEl) {
        recsListEl.innerHTML = '';
        data.recommendations.forEach(rec => {
            const item = document.createElement('div');
            item.className = `rec-item ${rec.type}`;
            item.innerHTML = `
                <div class="rec-icon">
                    <i class="fa-solid ${rec.icon}"></i>
                </div>
                <div>
                    <div class="rec-title">${escapeHtml(rec.title)}</div>
                    <div class="rec-text">${escapeHtml(rec.message)}</div>
                </div>
            `;
            recsListEl.appendChild(item);
        });
    }
}

function animateCounter(el, start, end, duration) {
    if (!el) return;
    const startTime = performance.now();
    const update = (now) => {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1.0);
        const current = start + (end - start) * easeOutQuad(progress);
        el.textContent = current.toFixed(1);
        if (progress < 1.0) {
            requestAnimationFrame(update);
        } else {
            el.textContent = end.toFixed(1);
        }
    };
    requestAnimationFrame(update);
}

function easeOutQuad(x) {
    return 1 - (1 - x) * (1 - x);
}

/* -------------------------------------------------------------
   MODEL PERFORMANCE CHARTS (performance.html)
   ------------------------------------------------------------- */
async function initPerformanceCharts() {
    try {
        const res = await fetch('/api/model-performance');
        const meta = await res.json();
        if (!meta || !meta.actual_vs_predicted) return;

        // 1. Actual vs Predicted Chart
        const ctxAvsP = document.getElementById('chartActualVsPred').getContext('2d');
        const avp = meta.actual_vs_predicted;
        const sampleLabels = avp.actual.map((_, i) => `#${i + 1}`);

        new Chart(ctxAvsP, {
            type: 'line',
            data: {
                labels: sampleLabels,
                datasets: [
                    {
                        label: 'Actual Test Score',
                        data: avp.actual,
                        borderColor: '#0f172a',
                        backgroundColor: 'rgba(15, 23, 42, 0.05)',
                        borderWidth: 2,
                        pointRadius: 4,
                        pointBackgroundColor: '#0f172a',
                        tension: 0.15
                    },
                    {
                        label: 'Predicted Score (Random Forest)',
                        data: avp.predicted,
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37, 99, 235, 0.08)',
                        borderWidth: 2,
                        pointRadius: 4,
                        pointBackgroundColor: '#2563eb',
                        borderDash: [4, 4],
                        tension: 0.15
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top' },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'Exam Score (0 - 100)' },
                        min: 30,
                        max: 100
                    },
                    x: {
                        title: { display: true, text: 'Test Samples' }
                    }
                }
            }
        });

        // 2. Feature Importance Horizontal Bar Chart
        const ctxFeat = document.getElementById('chartFeatureImportance').getContext('2d');
        const feats = meta.feature_importance || [];
        const featLabels = feats.map(f => f.label);
        const featVals = feats.map(f => f.importance);

        new Chart(ctxFeat, {
            type: 'bar',
            data: {
                labels: featLabels,
                datasets: [{
                    label: 'Relative Importance (%)',
                    data: featVals,
                    backgroundColor: [
                        '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#38bdf8',
                        '#4f46e5', '#6366f1', '#818cf8', '#a855f7', '#c084fc', '#e879f9'
                    ],
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Contribution: ${ctx.parsed.x}%`
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Importance Percentage (%)' },
                        beginAtZero: true
                    }
                }
            }
        });

        // 3. Model Comparison Chart (MAE, RMSE, R²)
        const ctxComp = document.getElementById('chartModelComparison').getContext('2d');
        const comp = meta.model_comparison_charts;

        new Chart(ctxComp, {
            type: 'bar',
            data: {
                labels: comp.labels,
                datasets: [
                    {
                        label: 'R² Score (Higher is Better)',
                        data: comp.r2_scores.map(s => s * 10), // Scale up for visual comparison
                        backgroundColor: '#10b981',
                        borderRadius: 6
                    },
                    {
                        label: 'MAE (Lower is Better)',
                        data: comp.mae_scores,
                        backgroundColor: '#3b82f6',
                        borderRadius: 6
                    },
                    {
                        label: 'RMSE (Lower is Better)',
                        data: comp.rmse_scores,
                        backgroundColor: '#f59e0b',
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: function(ctx) {
                                if (ctx.datasetIndex === 0) {
                                    return `R² Score: ${(ctx.parsed.y / 10).toFixed(4)}`;
                                }
                                return `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(3)}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Metric Magnitude (R² normalized x10)' }
                    }
                }
            }
        });

    } catch (err) {
        console.error('Error loading performance charts:', err);
    }
}

/* -------------------------------------------------------------
   STUDENT INSIGHTS CHARTS (insights.html)
   ------------------------------------------------------------- */
async function initInsightsCharts() {
    try {
        const res = await fetch('/api/insights');
        const data = await res.json();
        if (!data || !data.category_distribution) return;

        // 1. Performance Category Doughnut Chart
        const ctxDoughnut = document.getElementById('chartCategoryDoughnut').getContext('2d');
        const catMap = data.category_distribution;
        const catKeys = ['Excellent', 'Good', 'Average', 'At Risk'];
        const catVals = catKeys.map(k => catMap[k] || 0);

        new Chart(ctxDoughnut, {
            type: 'doughnut',
            data: {
                labels: catKeys,
                datasets: [{
                    data: catVals,
                    backgroundColor: ['#10b981', '#3b82f6', '#f59e0b', '#ef4444'],
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                },
                cutout: '65%'
            }
        });

        // 2. Attendance Distribution Bar Chart
        const ctxAtt = document.getElementById('chartAttendanceDist').getContext('2d');
        const attData = data.attendance_distribution;

        new Chart(ctxAtt, {
            type: 'bar',
            data: {
                labels: Object.keys(attData),
                datasets: [{
                    label: 'Students Count',
                    data: Object.values(attData),
                    backgroundColor: '#2563eb',
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, title: { display: true, text: 'Student Count' } },
                    x: { title: { display: true, text: 'Attendance Range' } }
                }
            }
        });

        // 3. Study Hours Distribution Bar Chart
        const ctxStudy = document.getElementById('chartStudyDist').getContext('2d');
        const studyData = data.study_hours_distribution;

        new Chart(ctxStudy, {
            type: 'bar',
            data: {
                labels: Object.keys(studyData),
                datasets: [{
                    label: 'Students Count',
                    data: Object.values(studyData),
                    backgroundColor: '#4f46e5',
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, title: { display: true, text: 'Student Count' } },
                    x: { title: { display: true, text: 'Study Hours/Day' } }
                }
            }
        });

        // 4. Backlogs vs Average Score Impact Line Chart
        const ctxBacklog = document.getElementById('chartBacklogImpact').getContext('2d');
        const backlogData = data.backlog_impact;

        new Chart(ctxBacklog, {
            type: 'line',
            data: {
                labels: Object.keys(backlogData).map(k => `${k} Backlogs`),
                datasets: [{
                    label: 'Average Final Score',
                    data: Object.values(backlogData),
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    fill: true,
                    borderWidth: 3,
                    pointRadius: 6,
                    pointBackgroundColor: '#ef4444',
                    tension: 0.2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { title: { display: true, text: 'Average Score' } }
                }
            }
        });

    } catch (err) {
        console.error('Error loading insights charts:', err);
    }
}

/* -------------------------------------------------------------
   HOME PREVIEW CHART (index.html)
   ------------------------------------------------------------- */
async function initHomePreviewChart() {
    try {
        const res = await fetch('/api/insights');
        const data = await res.json();
        const ctx = document.getElementById('homeCategoryChart');
        if (!ctx || !data.category_distribution) return;

        const catMap = data.category_distribution;
        const catKeys = ['Excellent', 'Good', 'Average', 'At Risk'];
        const catVals = catKeys.map(k => catMap[k] || 0);

        new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: catKeys,
                datasets: [{
                    label: 'Cohort Distribution',
                    data: catVals,
                    backgroundColor: ['#10b981', '#3b82f6', '#f59e0b', '#ef4444'],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, title: { display: true, text: 'Student Count' } }
                }
            }
        });
    } catch (err) {
        console.error('Home chart error:', err);
    }
}

/* -------------------------------------------------------------
   HISTORY MANAGEMENT (history.html)
   ------------------------------------------------------------- */
function initHistoryControls() {
    // Delete Single Record
    document.querySelectorAll('.btn-delete-pred').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const id = btn.dataset.id;
            if (!confirm(`Delete prediction record #${id}?`)) return;

            try {
                const res = await fetch(`/prediction-history/${id}`, { method: 'DELETE' });
                const json = await res.json();
                if (json.success) {
                    const row = document.getElementById(`pred-row-${id}`);
                    if (row) row.remove();
                    showToast(`Prediction #${id} deleted.`, 'info');
                } else {
                    showToast('Failed to delete record.', 'danger');
                }
            } catch (err) {
                console.error('Delete error:', err);
            }
        });
    });

    // Clear All Records
    const clearBtn = document.getElementById('btnClearAllHistory');
    if (clearBtn) {
        clearBtn.addEventListener('click', async () => {
            if (!confirm('Are you sure you want to permanently clear all prediction history?')) return;

            try {
                const res = await fetch('/prediction-history', { method: 'DELETE' });
                const json = await res.json();
                if (json.success) {
                    showToast('Prediction history cleared.', 'info');
                    setTimeout(() => window.location.reload(), 600);
                }
            } catch (err) {
                console.error('Clear history error:', err);
            }
        });
    }

    // History Table Live Filter
    const searchInput = document.getElementById('historySearchInput');
    const categorySelect = document.getElementById('historyCategorySelect');
    if (searchInput || categorySelect) {
        const filterRows = () => {
            const q = (searchInput ? searchInput.value : '').toLowerCase();
            const cat = categorySelect ? categorySelect.value : '';

            document.querySelectorAll('.history-table-row').forEach(row => {
                const name = (row.dataset.name || '').toLowerCase();
                const rowCat = row.dataset.category || '';
                const matchName = !q || name.includes(q);
                const matchCat = !cat || rowCat === cat;

                row.style.display = (matchName && matchCat) ? '' : 'none';
            });
        };

        if (searchInput) searchInput.addEventListener('input', filterRows);
        if (categorySelect) categorySelect.addEventListener('change', filterRows);
    }
}

/* -------------------------------------------------------------
   TOAST NOTIFICATION HELPER
   ------------------------------------------------------------- */
function showToast(message, type = 'info') {
    let container = document.getElementById('toastContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toastContainer';
        container.style.cssText = 'position: fixed; bottom: 24px; right: 24px; z-index: 9999; display: flex; flex-direction: column; gap: 8px;';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    const bgColors = {
        success: '#10b981',
        danger: '#ef4444',
        warning: '#f59e0b',
        info: '#2563eb'
    };

    toast.style.cssText = `
        background: ${bgColors[type] || bgColors.info};
        color: white;
        padding: 12px 18px;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        gap: 10px;
        opacity: 0;
        transform: translateY(10px);
        transition: all 0.25s ease;
    `;

    toast.innerHTML = `<i class="fa-solid fa-circle-info"></i> <span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '1';
        toast.style.transform = 'translateY(0)';
    }, 10);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        setTimeout(() => toast.remove(), 250);
    }, 3500);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
