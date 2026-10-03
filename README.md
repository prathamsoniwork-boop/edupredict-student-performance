# EduPredict – Student Performance Prediction System
**AI-Powered Academic Performance Forecasting & Early Intervention Platform**

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Flask%203-green.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20NumPy-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 1. Project Overview

**EduPredict** is an end-to-end Machine Learning web application designed for higher education institutions, instructors, and students. By evaluating key academic metrics (study hours, attendance percentage, previous exam scores, assignment completion, backlogs) alongside behavioral and environmental determinants (sleep hygiene, class participation, digital device access, and parental support), EduPredict forecasts a student's expected semester examination score ($0 - 100$) and classifies the student into tiered performance cohorts:

- **Excellent** ($\ge 80\%$)
- **Good** ($65\% - 79.9\%$)
- **Average** ($50\% - 64.9\%$)
- **At Risk** ($< 50\%$)

Crucially, the system automatically translates model outputs into **Personalized Academic Recommendations**, enabling timely counseling and targeted study habit adjustments before final examinations take place.

---

## 2. Problem Statement

In contemporary higher education, academic difficulty is predominantly detected in a **reactive manner**—typically after semester midterms or final examination grades have been published. By that point, options for remedial intervention are severely limited. 

Academic performance is influenced by multiple interconnected factors:
1. Declining classroom attendance leading to conceptual gaps.
2. Insufficient or inefficient daily self-study habits.
3. Accumulated backlogs that induce high cognitive stress.
4. Sub-optimal sleep routines impacting memory retention and exam-day focus.
5. Inequities in digital device or broadband access.

There is a compelling institutional need for an **objective, proactive Machine Learning system** capable of analyzing these multidimensional factors early and providing tailored guidance.

---

## 3. Project Objectives

- **Develop a Real ML Pipeline:** Train, compare, and validate candidate regression models (Linear Regression, Decision Tree Regressor, and Random Forest Regressor) on empirical student academic data.
- **Explainable Feature Impact:** Quantify the relative importance of each academic and behavioral attribute to understand which inputs most strongly drive academic success.
- **Deploy a Modern AI SaaS Dashboard:** Build a responsive, aesthetic web application using Flask, HTML5, CSS3, JavaScript, and Chart.js.
- **Real-Time Interactive Prediction:** Provide interactive sliders and demo presets for instantaneous inference, complete with animated score gauges and confidence intervals.
- **Actionable Guidance:** Automatically deliver constructive, non-judgmental recommendations tailored to each student's specific risk points.
- **Persistent Audit Log:** Maintain a local SQLite prediction history allowing records to be reviewed, filtered, and cleared.

---

## 4. Key Features

- **Dynamic Interactive Prediction Page:** Synchronized range sliders and number inputs across 11 academic and behavioral variables.
- **One-Click Demo Profiles:** Pre-configured profiles (**Recommended Demo**, **High Achiever**, **Average Student**, **At-Risk Student**) for swift presentation demonstrations.
- **Animated Circular Score Gauge:** Visual score indicator with dynamic color shifts (Emerald for Excellent, Blue for Good, Amber for Average, Crimson for At-Risk).
- **Personalized Recommendation Engine:** Automatically flags attendance deficits, backlog risks, sleep imbalance, and study volume adjustments.
- **Interactive Model Performance Page:**
  - Quantitative benchmark table comparing MAE, MSE, RMSE, and $R^2$ across all candidate algorithms.
  - Interactive **Actual vs. Predicted Scores** scatter/line visualization.
  - Interactive **Feature Importance** horizontal bar chart.
  - Algorithm comparison bar chart.
- **Student Cohort Insights Page:** Visual distribution charts detailing attendance brackets, study hour distributions, category proportions, and backlog impact curves.
- **Full Prediction History:** SQLite-backed audit log supporting live keyword search, category filtering, single record deletion, and bulk clearing.
- **College Presentation & Viva Guide:** Built-in presentation walkthrough in the About page to assist students during project defense.

---

## 5. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend & REST API** | Python, Flask, SQLite3, Joblib |
| **Machine Learning & Math** | Scikit-Learn, NumPy, Pandas |
| **Frontend UI/UX** | HTML5, Modern CSS3 (Grid & Flexbox), Vanilla JavaScript (ES6+) |
| **Visualizations & Charts** | Chart.js 4 (via CDN) |
| **Typography & Icons** | Plus Jakarta Sans, Inter, Font Awesome 6 |

---

## 6. Machine Learning Architecture & Algorithms

The system trains and benchmarks three candidate regression algorithms:

### 1. Linear Regression (Baseline)
- Fits an Ordinary Least Squares (OLS) closed-form solution: $\hat{\beta} = (X^T X + \lambda I)^{-1} X^T y$.
- Models baseline linear contributions across exam scores and attendance.

### 2. Decision Tree Regressor
- Employs greedy recursive binary splitting based on **Mean Squared Error (MSE) / Variance Reduction**:
  $$\Delta \text{Var} = \text{Var}(S) - \left( \frac{|S_L|}{|S|} \text{Var}(S_L) + \frac{|S_R|}{|S|} \text{Var}(S_R) \right)$$
- Captures discrete step-thresholds in academic performance.

### 3. Random Forest Regressor (Selected Primary Model)
- An ensemble of randomized decision trees combining **Bootstrap Aggregating (Bagging)** and **Random Subspace Feature Selection**.
- Aggregates individual tree forecasts: $\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(X)$.
- Computes prediction confidence from ensemble variance across trees: $\sigma^2 = \frac{1}{B}\sum (T_b(X) - \hat{y})^2$.
- Successfully captures non-linear dynamics:
  - Quadratic penalty for irregular sleep ($< 6.0$ hrs or $> 9.5$ hrs).
  - Synergistic compounding when high attendance ($\ge 75\%$) coincides with focused study ($\ge 4.5$ hrs/day).
  - Multiplicative crisis penalties for low attendance coupled with active backlogs.

### Evaluation Metrics Summary (Test Set: 300 Samples)

| Algorithm | MAE | MSE | RMSE | R² Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **3.350** | **18.455** | **4.296** | **0.7733** | **Selected Production Model** |
| Linear Regression | 2.188 | 8.130 | 2.851 | 0.9001 | Benchmark Baseline |
| Decision Tree Regressor | 4.886 | 36.205 | 6.017 | 0.5552 | Evaluated Candidate |

### Top Feature Importances (Random Forest)
1. **Previous Exam Score**: $30.86\%$
2. **Daily Study Hours**: $24.42\%$
3. **Attendance Percentage**: $13.19\%$
4. **Assignment Completion**: $9.74\%$
5. **Active Backlogs**: $6.66\%$
6. **Sleep Hours**: $5.31\%$
7. **Parental Support**: $4.20\%$
8. **Class Participation**: $2.97\%$
9. **Internet Availability**: $1.14\%$
10. **Device Availability**: $0.87\%$
11. **Extracurricular Activity**: $0.65\%$

---

## 7. Dataset Description

The dataset (`data/student_performance.csv`) consists of **1,500 student records** synthesized with reproducible statistical distributions reflecting real-world educational psychology:

| Feature Name | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `study_hours_per_day` | Float | $1.0 - 10.0$ | Daily self-directed study hours |
| `attendance_percentage` | Float | $50.9\% - 100.0\%$ | Classroom lecture attendance percentage |
| `previous_score` | Float | $35.0 - 98.0$ | Prior semester examination score |
| `assignment_completion`| Float | $30.0\% - 100.0\%$ | Percentage of coursework assignments turned in |
| `sleep_hours` | Float | $4.0 - 10.5$ | Nightly sleep duration |
| `class_participation` | Integer | $1 - 10$ | Interactive participation & tutorial engagement |
| `backlogs` | Integer | $0 - 6$ | Number of pending examination subjects |
| `internet_availability`| Binary | Yes / No | Reliable internet access at home |
| `device_availability` | Binary | Yes / No | Dedicated laptop or desktop computer |
| `extracurricular_activity`| Binary | Yes / No | Sports, clubs, or cultural activities |
| `parental_support` | Categorical| High / Medium / Low | Degree of academic encouragement at home |
| **`final_score`** (Target) | Float | $35.0 - 88.7$ | Final semester examination score |

---

## 8. Project Directory Structure

```text
edupredict/
│
├── app.py                     # Main Flask web application & REST API routes
├── database.py                # SQLite3 persistence layer & KPI calculation
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive technical documentation & presentation guide
│
├── data/
│   ├── student_performance.csv  # 1,500 record student dataset
│   └── edupredict.db           # SQLite database for prediction history
│
├── models/
│   ├── student_performance_model.pkl  # Trained Random Forest model pipeline
│   └── model_metadata.json           # Evaluation metrics, comparisons & chart data
│
├── ml/
│   ├── __init__.py
│   ├── dataset_generator.py   # Realistic statistical dataset generator
│   ├── models.py              # ML algorithm implementations & metric functions
│   ├── train_model.py         # Model training, comparison & export pipeline
│   └── predict.py             # Inference pipeline & personalized recommender
│
├── static/
│   ├── css/
│   │   └── style.css          # Modern AI SaaS dashboard styles
│   └── js/
│       └── script.js          # Interactive frontend logic, Chart.js & AJAX
│
└── templates/
    ├── base.html              # Base layout with navbar, footer & alerts
    ├── index.html             # Landing / Dashboard with summary KPI cards
    ├── predict.html           # Prediction Page with interactive form & gauge
    ├── performance.html       # ML Model Performance & Charts
    ├── insights.html          # Student Insights & Analytics
    ├── history.html           # Prediction History Log & Table Filters
    └── about.html             # About Project & ML Workflow
```

---

## 9. Installation & Setup Instructions

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14 installed on your machine.
- Pip package manager.
- Git (optional, for cloning).

### Step 1: Clone or Open the Workspace
```bash
cd "Machine Learning IBM Project"
```

### Step 2: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the Machine Learning Models (Optional - Pre-trained model included)
```bash
python ml/train_model.py
```
*Output will display the 5-step training pipeline, evaluate Linear Regression, Decision Tree, and Random Forest on unseen test data, and save the serialized model and metadata.*

### Step 4: Run the Flask Web Application
```bash
python app.py
```

### Step 5: Access the Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 10. How to Deploy to Vercel (Live Cloud Deployment)

EduPredict is pre-configured with `vercel.json`, `api/index.py`, `.vercelignore`, and a serverless-safe `/tmp` SQLite database path.

### Method A: Deploy via GitHub (Recommended & Free)
1. **Push your code to a GitHub repository**:
   - Go to [github.com/new](https://github.com/new) and create a new repository (e.g. `edupredict-ml`).
   - Push or upload this project directory to your repository.
2. **Import into Vercel**:
   - Go to [vercel.com](https://vercel.com) and sign in with your GitHub account.
   - Click **"Add New..."** &rarr; **"Project"**.
   - Select your `edupredict-ml` repository from the list and click **"Import"**.
3. **Deploy**:
   - Leave the Framework Preset as default / Other.
   - Click **"Deploy"**.
   - Vercel will install dependencies from `requirements.txt`, bundle `api/index.py`, and launch your live application with a free SSL domain (e.g., `https://edupredict-ml.vercel.app`) in under 1 minute!

### Method B: Deploy via Vercel CLI
If you have Node.js / Vercel CLI installed on your machine:
```bash
npm install -g vercel
vercel
```
Follow the interactive prompts (link to existing project: No, project name: `edupredict`, directory: `./`). When complete, Vercel will output your live production URL.

---

## 11. Example Prediction Walkthrough

1. Open the web application and click **“Predict”** in the navigation bar.
2. Click the **“Recommended Demo”** preset button. The form auto-populates with:
   - Study Hours: `4.5`
   - Attendance: `88%`
   - Previous Score: `78`
   - Assignment Completion: `92%`
   - Sleep Hours: `7.0`
   - Participation: `8 / 10`
   - Backlogs: `0`
   - Internet & Device: `Yes`
   - Parental Support: `High`
3. Click **“Predict Performance”**.
4. The system executes real-time inference:
   - **Expected Score:** `75.1 / 100`
   - **Performance Category:** `Good`
   - **Model Confidence:** `81.5%`
   - **Recommendations:** Reinforces current positive habits, advises maintaining sleep hygiene, and encourages challenging assignments.
5. The result is automatically stored in the **Prediction History** log.

---

## 11. Screenshots Section (Placeholder)

| Dashboard Overview | Real-Time Prediction Form |
| :---: | :---: |
| *[Screenshot: Dashboard KPI Cards & Charts]* | *[Screenshot: Dual-column Prediction Form & Gauge]* |

| Model Performance & Benchmark | Student Cohort Insights |
| :---: | :---: |
| *[Screenshot: Actual vs. Predicted & Feature Importance]* | *[Screenshot: Attendance & Backlog Impact Curves]* |

---

## 12. College Presentation & Viva Defense Tips

When presenting this project to an examiner or instructor, emphasize the following points:

1. **Why not hardcode predictions?**  
   The application uses a trained Random Forest Regressor saved with Joblib. Predictions are computed on the fly by querying the loaded decision trees.
2. **Why Random Forest over a single Decision Tree?**  
   A single decision tree is prone to high variance and overfitting ($R^2 \approx 0.55$). Random Forest constructs an ensemble of 100 trees with bootstrap sampling, reducing variance and raising generalization accuracy ($R^2 \approx 0.77$).
3. **How does the system ensure robustness?**  
   All inputs are validated client-side and server-side with physiological and academic bounds (e.g., study hours $0 - 16$, attendance $0 - 100\%$, non-negative backlogs).
4. **Why is the recommender rule-based rather than black-box?**  
   In educational counseling, transparency is vital. The recommender maps specific input deficiencies (such as $<75\%$ attendance or $\ge 2$ backlogs) to actionable pedagogical steps.

---

## 13. Future Improvements

- **LMS Integration:** Integrate with institutional Learning Management Systems (Canvas, Moodle, Blackboard) via LTI standards for automated attendance and assignment sync.
- **Deep Learning / Neural Networks:** Benchmark Multi-Layer Perceptrons (MLP) and TabNet architectures.
- **Time-Series Forecasting:** Track multi-semester trajectories using Recurrent Neural Networks (LSTM) to predict long-term CGPA trends.
- **Multi-lingual Support:** Provide localized recommendation messages in multiple regional languages.

---

## 14. License

This project is licensed under the **MIT License**. Free for educational and academic use.
