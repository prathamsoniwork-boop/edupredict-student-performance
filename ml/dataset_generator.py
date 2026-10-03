"""
EduPredict Dataset Generator
Generates a realistic, statistically grounded student academic performance dataset (1,500 records).
Models real educational dynamics with non-linear factors, interactions, and reproducible random seed.
"""

import numpy as np
import pandas as pd
import os

def generate_student_dataset(n_samples=1500, random_seed=42):
    """
    Generate realistic student academic data with 11 features and final score.
    """
    np.random.seed(random_seed)
    
    # 1. Study Hours Per Day (Normal distribution bounded between 1.0 and 10.0)
    study_hours = np.clip(np.random.normal(loc=4.2, scale=1.8, size=n_samples), 1.0, 10.0)
    study_hours = np.round(study_hours, 1)

    # 2. Attendance Percentage (Beta-like distribution skewed towards higher attendance)
    raw_attendance = np.random.beta(a=5, b=2, size=n_samples) * 60 + 40
    attendance = np.clip(np.round(raw_attendance, 1), 40.0, 100.0)

    # 3. Previous Exam Score (Correlated partially with baseline academic skill)
    base_ability = np.random.normal(loc=70, scale=14, size=n_samples)
    previous_score = np.clip(np.round(base_ability + np.random.normal(0, 5, n_samples), 1), 35.0, 98.0)

    # 4. Assignment Completion Percentage (Related to attendance and conscientiousness)
    raw_assignments = (0.5 * attendance + 0.3 * (study_hours * 10) + np.random.normal(15, 8, n_samples))
    assignment_completion = np.clip(np.round(raw_assignments, 1), 30.0, 100.0)

    # 5. Sleep Hours (Bell-shaped around 7 hours per night)
    sleep_hours = np.clip(np.round(np.random.normal(loc=7.1, scale=1.2, size=n_samples), 1), 4.0, 10.5)

    # 6. Class Participation Score (Integer scale 1 to 10)
    raw_part = (0.05 * attendance + 0.3 * study_hours + np.random.normal(2, 1.5, n_samples))
    class_participation = np.clip(np.round(raw_part).astype(int), 1, 10)

    # 7. Number of Previous Backlogs (Poisson distribution, most students have 0, some 1-4)
    backlogs = np.random.poisson(lam=0.7, size=n_samples)
    backlogs = np.clip(backlogs, 0, 6)

    # 8. Internet Availability (Binary: Yes / No - 85% Yes)
    internet_p = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.88, 0.12])

    # 9. Device Availability (Binary: Yes / No - 82% Yes)
    device_p = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.84, 0.16])

    # 10. Extracurricular Activity (Binary: Yes / No - 60% Yes)
    extracurricular_p = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.60, 0.40])

    # 11. Parental/Guardian Support (Categorical: High, Medium, Low)
    parental_p = np.random.choice(['High', 'Medium', 'Low'], size=n_samples, p=[0.42, 0.43, 0.15])

    # ---- Synthesize Final Academic Score with Realistic Educational Relationships ----
    # Base formula weights reflecting real pedagogical research:
    # - Previous Score: strong predictor (~32% weight)
    # - Attendance: strong predictor (~20% weight)
    # - Study Hours: high impact (~18% weight)
    # - Assignment Completion: ~12% weight
    # - Class Participation: ~6% weight
    # - Backlogs penalty: -3.0 points per active backlog
    # - Sleep curve: optimal between 6.8 - 8.2 hrs. Penalty for <5.5 hrs (fatigue) or >9.5 hrs (lethargy)
    # - Digital access: Internet (+2.0), Device (+2.2)
    # - Extracurriculars: (+1.5)
    # - Parental support: High (+3.2), Medium (+0.8), Low (-2.5)
    # - Interactive boost: High study hours + high attendance yields synergy

    # Linear components
    score = (
        0.32 * previous_score +
        0.20 * attendance +
        1.80 * study_hours +
        0.12 * assignment_completion +
        0.70 * class_participation
    )

    # Backlog penalty
    score -= (3.2 * backlogs)

    # Sleep curve (quadratic penalty centered at 7.5 hours)
    # Peak at 7.5; 5 hours gives penalty -(2.5)^2 * 0.8 = -5.0 points; 10 hours gives -(2.5)^2 * 0.8 = -5.0
    sleep_penalty = -0.75 * ((sleep_hours - 7.5) ** 2)
    score += sleep_penalty

    # Categorical bonuses
    internet_bonus = np.where(internet_p == 'Yes', 2.2, -1.8)
    device_bonus = np.where(device_p == 'Yes', 2.4, -1.5)
    extra_bonus = np.where(extracurricular_p == 'Yes', 1.5, 0.0)
    
    parental_bonus = np.zeros(n_samples)
    parental_bonus[parental_p == 'High'] = 3.5
    parental_bonus[parental_p == 'Medium'] = 0.8
    parental_bonus[parental_p == 'Low'] = -2.8

    score += internet_bonus + device_bonus + extra_bonus + parental_bonus

    # Synergy interaction: students with both >75% attendance AND >5 hrs study do exceptionally well
    synergy = np.where((attendance >= 75) & (study_hours >= 4.5), 2.5, 0.0)
    # Compounding risk: low attendance (<60) and backlogs (>1) leads to severe academic crisis
    crisis_penalty = np.where((attendance < 60) & (backlogs >= 2), -4.5, 0.0)

    score += synergy + crisis_penalty

    # Add realistic stochastic noise (exam day variance, random variations)
    noise = np.random.normal(0, 2.2, size=n_samples)
    final_score = np.clip(np.round(score + noise, 1), 20.0, 99.5)

    df = pd.DataFrame({
        'student_id': [f'STU{1000 + i}' for i in range(n_samples)],
        'study_hours_per_day': study_hours,
        'attendance_percentage': attendance,
        'previous_score': previous_score,
        'assignment_completion': assignment_completion,
        'sleep_hours': sleep_hours,
        'class_participation': class_participation,
        'backlogs': backlogs,
        'internet_availability': internet_p,
        'device_availability': device_p,
        'extracurricular_activity': extracurricular_p,
        'parental_support': parental_p,
        'final_score': final_score
    })

    return df

def save_dataset(output_path='data/student_performance.csv', n_samples=1500):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_student_dataset(n_samples=n_samples, random_seed=42)
    df.to_csv(output_path, index=False)
    print(f"Generated realistic dataset with {len(df)} records at: {output_path}")
    print("Dataset Summary Statistics:")
    print(df[['study_hours_per_day', 'attendance_percentage', 'previous_score', 'final_score']].describe())
    return df

if __name__ == '__main__':
    save_dataset()
