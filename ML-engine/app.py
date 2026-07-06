import flask
from flask_cors import CORS
from flask import request, jsonify
import pandas as pd
import pickle
import sqlite3
from datetime import datetime

app = flask.Flask(__name__, static_folder='../dashboard', static_url_path='')
CORS(app)

# ── Ensure the prediction_history table exists ──────────────────────────────
def _init_history_table():
    conn = sqlite3.connect('saltguard.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            patient_label TEXT,
            anchor_age    REAL,
            gender        TEXT,
            Creatinine    REAL,
            Hemoglobin    REAL,
            Sodium        REAL,
            "Urea Nitrogen" REAL,
            WBC           REAL,
            mortality_probability REAL,
            prediction_class      INTEGER
        )
    ''')
    conn.commit()
    conn.close()

_init_history_table()

@app.route('/')
def home():
    return flask.send_from_directory(app.static_folder, 'login.html')

# Load model and feature columns
# IMPORTANT: these pickles must be loaded from the same working directory as this file.
# If the API is started from a different folder, relative paths will break.
try:
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "xgb_mortality_model.pkl")
    features_path = os.path.join(base_dir, "xgb_feature_columns.pkl")

    print("[model] loading:", model_path)
    print("[features] loading:", features_path)
    model = pickle.load(open(model_path, "rb"))
    features = pickle.load(open(features_path, "rb"))
    print("[features] loaded count:", len(features))
    print("[features] first:", list(features)[:20])
except Exception as e:
    # Fail fast so /predict does not crash with 'features' undefined.
    raise RuntimeError(f"Could not load model or feature columns: {e}")



@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features and calculate the derived variables based on standard rules
        age = float(data.get("anchor_age", 60))
        # Gender is neutralized (fixed at 0.5) so that predictions are driven
        # purely by clinical lab values and not by the gender bias the model
        # learned from the imbalanced training dataset.
        gender_mapped = 0.5
        
        creat = float(data.get("Creatinine", 1.0))
        hemo = float(data.get("Hemoglobin", 12.0))
        sodium = float(data.get("Sodium", 140.0))
        urea = float(data.get("Urea Nitrogen", 15.0))
        wbc = float(data.get("WBC", 7.0))
        
        # Engineered features as per ML logic
        bun_creatinine_ratio = (urea / creat) if creat > 0 else 0
        renal_dysfunction_flag = 1 if (creat >= 1.5) else 0
        leukocytosis = 1 if (wbc > 11.0) else 0
        elderly_risk = 1 if (age >= 60) else 0
        severe_anemia = 1 if (hemo < 9.0) else 0
        hyponatremia = 1 if (sodium < 135.0) else 0
        severe_hyponatremia = 1 if (sodium < 130.0) else 0
        
        row = {
            'anchor_age': age,
            'gender': gender_mapped,
            'Creatinine': creat,
            'Hemoglobin': hemo,
            'Sodium': sodium,
            'Urea Nitrogen': urea,
            'WBC': wbc,
            'bun_creatinine_ratio': bun_creatinine_ratio,
            'renal_dysfunction_flag': renal_dysfunction_flag,
            'severe_anemia': severe_anemia,
            'hyponatremia': hyponatremia,
            'severe_hyponatremia': severe_hyponatremia,
            'leukocytosis': leukocytosis,
            'elderly_risk': elderly_risk
        }
        
        df = pd.DataFrame([row])
        # Reorder to match model expectations
        df = df[features]
        
        # Predict probability
        prob = model.predict_proba(df)[0][1] * 100
        pred_class = int(model.predict(df)[0])
        
        # ── Persist to prediction_history ───────────────────────────────────
        gender_label = data.get("gender", "M")
        patient_label = data.get("patient_label", "Manual Entry")
        conn = sqlite3.connect('saltguard.db')
        conn.execute('''
            INSERT INTO prediction_history
              (timestamp, patient_label, anchor_age, gender,
               Creatinine, Hemoglobin, Sodium, "Urea Nitrogen", WBC,
               mortality_probability, prediction_class)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            patient_label,
            age, gender_label,
            creat, hemo, sodium, urea, wbc,
            round(float(prob), 1), pred_class
        ))
        conn.commit()
        conn.close()

        # Return probability and also scaled lab indicators for dashboard UI
        return jsonify({
            "status": "success",
            "mortality_probability": round(float(prob), 1),
            "prediction_class": pred_class,
            "inputs": {
                "Creatinine": creat,
                "Sodium": sodium,
                "WBC": wbc,
                "Urea_Nitrogen": urea,
                "Hemoglobin": hemo
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/patients', methods=['GET'])
def get_patients():
    try:
        conn = sqlite3.connect('saltguard.db')
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        query = '''
            SELECT f.hadm_id, f.subject_id, f.anchor_age, f.gender, f.Creatinine,
                   f.Hemoglobin, f.Sodium, f."Urea Nitrogen", f.WBC, p.anchor_year_group
            FROM final_features f
            JOIN patients p ON f.subject_id = p.subject_id
            LIMIT 50
        '''
        cur.execute(query)
        rows = cur.fetchall()
        patients = [dict(row) for row in rows]
        conn.close()
        return jsonify({"status": "success", "data": patients})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/history', methods=['GET', 'DELETE'])
def get_history():
    if flask.request.method == 'DELETE':
        try:
            conn = sqlite3.connect('saltguard.db')
            conn.execute('DELETE FROM prediction_history')
            conn.commit()
            conn.close()
            return jsonify({"status": "success", "message": "History cleared"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    # GET
    try:
        conn = sqlite3.connect('saltguard.db')
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute('''
            SELECT id, timestamp, patient_label, anchor_age, gender,
                   Creatinine, Hemoglobin, Sodium, "Urea Nitrogen", WBC,
                   mortality_probability, prediction_class
            FROM prediction_history
            ORDER BY id DESC
            LIMIT 100
        ''')
        rows = cur.fetchall()
        history = [dict(row) for row in rows]
        conn.close()
        return jsonify({"status": "success", "data": history})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/ai-insight', methods=['POST'])
def ai_insight():
    """
    Enhanced Clinical Decision Support engine v2.
    Adds: risk trend, explainability, alerts, differential dx,
    abnormal highlights, quick summary, what-if simulation.
    """
    try:
        data        = request.json
        age         = float(data.get('anchor_age', 60))
        gender      = data.get('gender', 'M')
        prob        = float(data.get('mortality_probability', 0))
        sodium      = float(data.get('Sodium', 140))
        creat       = float(data.get('Creatinine', 1.0))
        urea        = float(data.get('Urea Nitrogen', 15))
        hemo        = float(data.get('Hemoglobin', 12))
        wbc         = float(data.get('WBC', 7))
        patient_id  = data.get('patient_id', 'SG-LIVE')
        patient_name = data.get('patient_name', 'Anonymous')

        # ── Risk tier ──────────────────────────────────────────────────────
        if prob >= 70:
            risk_level  = 'High';   risk_color = 'danger';  urgency = 'Critical'
            risk_interp = ('Elevated severity probability indicating advanced hepatic decompensation '
                           'requiring immediate clinical intervention.')
        elif prob >= 30:
            risk_level  = 'Moderate'; risk_color = 'warning'; urgency = 'Moderate'
            risk_interp = ('Intermediate risk profile; close monitoring and proactive management '
                           'are essential to prevent clinical deterioration.')
        else:
            risk_level  = 'Low';    risk_color = 'normal';  urgency = 'Low'
            risk_interp = ('Lab values within acceptable range; routine monitoring is advised '
                           'with reassessment if symptoms change.')

        # ── Risk trend from prediction_history (last 3 records) ────────────
        trend = []
        try:
            conn_t = sqlite3.connect('saltguard.db')
            cur_t  = conn_t.cursor()
            cur_t.execute(
                'SELECT mortality_probability, timestamp FROM prediction_history ORDER BY id DESC LIMIT 3'
            )
            rows_t = cur_t.fetchall()
            conn_t.close()
            labels = ['Today', '3 Checks Ago', '6 Checks Ago']
            for i, (p, ts) in enumerate(rows_t):
                trend.append({'risk': round(float(p), 1), 'label': labels[i], 'timestamp': ts})
        except Exception:
            trend = []

        if len(trend) >= 2:
            delta = trend[0]['risk'] - trend[-1]['risk']
            if delta > 5:
                trend_interp = 'Risk is WORSENING — significant upward trend observed.'
                trend_dir    = 'worsening'
            elif delta < -5:
                trend_interp = 'Risk is IMPROVING — downward trend since last assessment.'
                trend_dir    = 'improving'
            else:
                trend_interp = 'Risk is STABLE — no significant change across assessments.'
                trend_dir    = 'stable'
        else:
            trend_interp = 'Insufficient history — first assessment recorded.'
            trend_dir    = 'unknown'

        # ── Abnormal value highlighting ────────────────────────────────────
        def lab_highlight(name, value, unit, low_crit, low_warn, high_warn, high_crit, normal_range, tooltip):
            if value <= low_crit or value >= high_crit:
                icon = '🔴'; sev = 'danger';  status = 'Critical'
            elif value <= low_warn or value >= high_warn:
                icon = '🟡'; sev = 'warning'; status = 'Abnormal'
            else:
                icon = '🟢'; sev = 'normal';  status = 'Normal'
            return {
                'name': name, 'value': value, 'unit': unit,
                'icon': icon, 'severity': sev, 'status': status,
                'normal_range': normal_range, 'tooltip': tooltip
            }

        bun_cr = round(urea / creat, 1) if creat > 0 else 0
        abnormal_labs = [
            lab_highlight('Sodium',          sodium, 'mEq/L', 125, 135, 145, 150,
                          '135–145 mEq/L',
                          'Hyponatremia drives fluid shifts, ascites, and encephalopathy in cirrhosis.'),
            lab_highlight('Creatinine',      creat,  'mg/dL', 0,   0,   1.2, 2.5,
                          '0.6–1.2 mg/dL',
                          'Rising creatinine signals renal dysfunction — key hepatorenal syndrome marker.'),
            lab_highlight('Urea Nitrogen',   urea,   'mg/dL', 0,   0,   25,  40,
                          '7–25 mg/dL',
                          f'BUN/Cr ratio {bun_cr} — elevated ratio suggests GI bleed or pre-renal azotaemia.'),
            lab_highlight('Hemoglobin',      hemo,   'g/dL',  7,   9,   16,  20,
                          '12–17 g/dL',
                          'Anaemia in cirrhosis is multifactorial: hypersplenism, GI bleed, or nutritional deficiency.'),
            lab_highlight('WBC Count',       wbc,    'K/µL',  2,   3,   11,  15,
                          '4–11 K/µL',
                          'Leukocytosis in cirrhosis warrants urgent exclusion of SBP or systemic infection.'),
        ]

        # ── Lab findings (narrative) ───────────────────────────────────────
        # ── Lab findings (narrative) ───────────────────────────────────────
        findings   = []
        lab_status = {
            'sodium': (abnormal_labs[0]['status'], abnormal_labs[0]['severity']),
            'creat':  (abnormal_labs[1]['status'], abnormal_labs[1]['severity']),
            'urea':   (abnormal_labs[2]['status'], abnormal_labs[2]['severity']),
            'hemo':   (abnormal_labs[3]['status'], abnormal_labs[3]['severity']),
            'wbc':    (abnormal_labs[4]['status'], abnormal_labs[4]['severity'])
        }
        
        if sodium < 125:
            findings.append(f'Severe hyponatremia (Na⁺ {sodium} mEq/L) — high risk of cerebral oedema and encephalopathy.')
        elif sodium < 135:
            findings.append(f'Hyponatremia (Na⁺ {sodium} mEq/L) — associated with ascites and hepatorenal syndrome risk.')

        if creat >= 2.5:
            findings.append(f'Severely elevated creatinine ({creat} mg/dL) — AKI / hepatorenal syndrome likely.')
        elif creat >= 1.2:
            findings.append(f'Elevated creatinine ({creat} mg/dL) — renal dysfunction; consider hepatorenal syndrome workup.')

        if urea > 40:
            findings.append(f'Elevated BUN ({urea} mg/dL, BUN/Cr ratio {bun_cr}) — pre-renal azotaemia or GI bleed suspected.')
        elif urea > 25:
            pass # narrative not strictly needed for borderline, but handled

        if hemo < 7:
            findings.append(f'Severe anaemia (Hgb {hemo} g/dL) — transfusion threshold met; assess for GI haemorrhage.')
        elif hemo < 9:
            findings.append(f'Anaemia (Hgb {hemo} g/dL) — likely multifactorial in cirrhosis; monitor haemodynamics.')

        if wbc > 15:
            findings.append(f'Marked leukocytosis (WBC {wbc} K/µL) — SBP or sepsis must be excluded immediately.')
        elif wbc > 11:
            findings.append(f'Leukocytosis (WBC {wbc} K/µL) — infectious or inflammatory process; consider paracentesis.')
        elif wbc < 3:
            findings.append(f'Leukopenia (WBC {wbc} K/µL) — hypersplenism or bone marrow suppression.')

        if not findings:
            findings.append('All key laboratory markers are within reference ranges.')

        # ── Risk Contribution / Explainability ────────────────────────────
        # Calculate raw deviations from normal ranges, with caps to prevent extreme outlier typos from dominating
        sod_dev = (135 - sodium) if sodium < 135 else (sodium - 145) if sodium > 145 else 0
        sod_raw = min(max(0, sod_dev * 4.0), 150)

        creat_raw = min(max(0, (creat - 1.2) * 22), 150)
        bun_raw   = min(max(0, (urea - 25) * 1.8), 150)
        
        hemo_dev = (12 - hemo) if hemo < 12 else (hemo - 17) if hemo > 17 else 0
        hemo_raw  = min(max(0, hemo_dev * 4.5), 150)

        wbc_dev = (wbc - 11) if wbc > 11 else (4 - wbc) if wbc < 4 else 0
        wbc_raw   = min(max(0, wbc_dev * 3.5), 150)

        total_raw = sod_raw + creat_raw + bun_raw + hemo_raw + wbc_raw

        if total_raw <= 0:
            contributions = { 'Sodium': 0, 'Creatinine': 0, 'BUN': 0, 'Hemoglobin': 0, 'WBC': 0 }
        else:
            def pct(x): return round(x / total_raw * 100)
            contributions = {
                'Sodium':     pct(sod_raw),
                'Creatinine': pct(creat_raw),
                'BUN':        pct(bun_raw),
                'Hemoglobin': pct(hemo_raw),
                'WBC':        pct(wbc_raw),
            }
            # Fix rounding to sum exactly 100
            diff = 100 - sum(contributions.values())
            if diff != 0:
                max_key = max(contributions, key=contributions.get)
                contributions[max_key] += diff

        sod_msg = 'Hyponatremia' if sodium < 135 else 'Hypernatremia' if sodium > 145 else 'Sodium abnormality'
        wbc_msg = 'Leukopenia' if wbc < 4 else 'Leukocytosis'
        hemo_msg = 'Anaemia' if hemo < 12 else 'High Hemoglobin'

        contrib_explanations = {
            'Sodium':     f'{sod_msg} disrupts fluid balance; Na⁺ {sodium} mEq/L driving {contributions["Sodium"]}% of risk.',
            'Creatinine': f'Renal dysfunction marker; creatinine {creat} mg/dL contributes {contributions["Creatinine"]}% of risk.',
            'BUN':        f'Azotaemia indicator; BUN {urea} mg/dL contributes {contributions["BUN"]}% of risk.',
            'Hemoglobin': f'{hemo_msg} impacts O₂ delivery; Hgb {hemo} g/dL contributes {contributions["Hemoglobin"]}% of risk.',
            'WBC':        f'{wbc_msg} signals infection/inflammation; WBC {wbc} K/µL contributes {contributions["WBC"]}% of risk.',
        }
        contribution_list = sorted(
            [{'param': k, 'pct': v, 'explanation': contrib_explanations[k],
              'severity': lab_status.get(k.lower().replace(' ', '').replace('bun','urea'), ('Normal','normal'))[1]}
             for k, v in contributions.items()],
            key=lambda x: -x['pct']
        )

        # ── Real-Time Alerts (only critical) ─────────────────────────────
        alerts = []
        if sodium < 125:
            alerts.append({'level': 'critical', 'icon': '🚨',
                           'message': 'Severe hyponatremia detected (Na⁺ < 125 mEq/L)',
                           'urgency': 'Risk of cerebral oedema — immediate electrolyte correction required.'})
        if creat >= 2.5:
            alerts.append({'level': 'critical', 'icon': '🚨',
                           'message': 'Acute kidney injury / hepatorenal syndrome suspected',
                           'urgency': 'Nephrology consult and IV fluid assessment needed without delay.'})
        if hemo < 7:
            alerts.append({'level': 'critical', 'icon': '🚨',
                           'message': 'Severe anaemia — transfusion threshold met',
                           'urgency': 'Crossmatch and transfuse; rule out active GI haemorrhage urgently.'})
        if wbc > 15:
            alerts.append({'level': 'critical', 'icon': '🚨',
                           'message': 'Marked leukocytosis — sepsis / SBP must be excluded',
                           'urgency': 'Blood + ascitic fluid cultures; empirical antibiotics without delay.'})
        if prob >= 70:
            alerts.append({'level': 'warning', 'icon': '⚠️',
                           'message': f'Cirrhosis severity risk {prob:.1f}% — ICU-level care may be indicated',
                           'urgency': 'Hepatology + nephrology multidisciplinary team review within 24 hours.'})

        # ── Complications ─────────────────────────────────────────────────
        complications = []
        if sodium < 135 or creat >= 1.5:
            complications.append('Hepatorenal Syndrome (HRS) — progressive renal failure in decompensated cirrhosis.')
        if sodium < 135:
            complications.append('Hepatic Encephalopathy — altered mentation secondary to hyperammonaemia.')
        if hemo < 9 or wbc > 11:
            complications.append('Spontaneous Bacterial Peritonitis (SBP) — life-threatening ascitic fluid infection.')
        if creat >= 1.5:
            complications.append('Acute Kidney Injury (AKI) — requires nephrology consult.')
        if prob >= 50:
            complications.append('Variceal Haemorrhage — elevated portal hypertension risk.')
        complications.append('Coagulopathy — impaired hepatic synthetic function.')
        complications = complications[:5]

        # ── Differential Diagnosis ────────────────────────────────────────
        differential = []
        if sodium < 135 or creat >= 1.5:
            differential.append({'condition': 'Hepatorenal Syndrome (HRS)',
                                 'probability': 'High' if (sodium < 130 and creat >= 1.5) else 'Moderate',
                                 'basis': 'Hyponatremia + elevated creatinine in cirrhosis context.'})
        if sodium < 135:
            differential.append({'condition': 'Hepatic Encephalopathy',
                                 'probability': 'Moderate',
                                 'basis': 'Electrolyte dysregulation with hyperammonaemia risk.'})
        if wbc > 11:
            differential.append({'condition': 'Spontaneous Bacterial Peritonitis (SBP)',
                                 'probability': 'High' if wbc > 15 else 'Moderate',
                                 'basis': 'Leukocytosis in decompensated cirrhosis — requires paracentesis.'})
        if hemo < 9:
            differential.append({'condition': 'Variceal Haemorrhage',
                                 'probability': 'Moderate',
                                 'basis': 'Anaemia with portal hypertension — rule out active bleed.'})
        if urea > 40:
            differential.append({'condition': 'Upper GI Bleed / Pre-renal Azotaemia',
                                 'probability': 'Moderate',
                                 'basis': f'BUN/Cr ratio {bun_cr} — disproportionate elevation suggests GI bleed.'})
        differential.append({'condition': 'Decompensated Liver Cirrhosis',
                             'probability': 'High' if prob >= 50 else 'Moderate',
                             'basis': 'Overall lab pattern consistent with hepatic decompensation.'})
        differential = differential[:5]

        # ── Recommended Actions ───────────────────────────────────────────
        actions = [
            'Obtain serum ammonia, PT/INR, albumin, bilirubin, and LFT panel urgently.',
            'Perform diagnostic paracentesis if ascites present (cell count + culture).',
        ]
        if sodium < 135:
            actions.append('Initiate fluid restriction; consider tolvaptan if Na⁺ < 130 mEq/L under specialist guidance.')
        if creat >= 1.5:
            actions.append('Discontinue nephrotoxic agents; assess volume status and hourly urine output.')
        if hemo < 9:
            actions.append('Crossmatch and transfuse if clinically indicated; upper GI endoscopy to rule out variceal bleed.')
        if wbc > 11:
            actions.append('Obtain blood, urine, and ascitic fluid cultures; initiate empirical antibiotics (cefotaxime 2 g IV q8h).')
        if prob >= 70:
            actions.append('ICU-level monitoring; hepatology + nephrology MDT review within 24 hours.')
        elif prob >= 30:
            actions.append('Admit for in-patient monitoring; reassess labs in 24–48 hours.')
        else:
            actions.append('Outpatient follow-up in 2–4 weeks; advise salt restriction and diuretic compliance.')

        # ── Confidence score ──────────────────────────────────────────────
        abnormal_count = sum(1 for v in lab_status.values() if v[1] != 'normal')
        confidence = min(72 + abnormal_count * 5, 96)

        # ── Urgency reason ────────────────────────────────────────────────
        urgency_reason_map = {
            'Critical': 'Multiple critical lab aberrations with high predicted severity.',
            'Moderate': 'Abnormal lab values with intermediate risk; requires close surveillance.',
            'Low':      'Lab values within acceptable range; no immediate intervention required.'
        }
        urgency_reason = urgency_reason_map[urgency]

        # ── Quick Doctor Summary (2 lines) ────────────────────────────────
        abnormal_names = [f.split('—')[0].strip() for f in findings[:2]] or ['No significant abnormalities']
        quick_summary = [
            (f'{int(age)}-year-old {("male" if gender=="M" else "female")} — '
             f'{risk_level.lower()} cirrhosis severity risk ({prob:.1f}%). '
             f'Key abnormalities: {"; ".join(abnormal_names)}.'),
            (f'Urgency: {urgency}. {actions[0]}')
        ]

        # ── What-if Simulation ────────────────────────────────────────────
        top_param = max(contributions, key=contributions.get)
        top_pct   = contributions[top_param]
        simulated_reduction = round(top_pct * 0.6)
        sim1_val = max(0, round(prob - simulated_reduction, 1))
        sim2_val = max(0, round(prob * 0.42, 1))

        param_targets = {
            'Sodium':     f'sodium normalises to ≥135 mEq/L',
            'Creatinine': f'creatinine reduces to <1.2 mg/dL with treatment',
            'BUN':        f'BUN normalises to <25 mg/dL',
            'Hemoglobin': f'Hgb corrects to ≥12 g/dL via transfusion',
            'WBC':        f'WBC resolves to <11 K/µL with antibiotic therapy',
        }
        whatif = [
            f'If {param_targets.get(top_param, "primary abnormality resolves")}, risk may reduce to ~{sim1_val}%.',
            f'If all abnormal values normalise with targeted therapy, risk could reduce to ~{sim2_val}%.',
        ]

        # ── MELD Score Calculation ─────────────────────────────────────────
        import math
        bili_val = max(1.0, float(data.get('Bilirubin', 1.2)))
        inr_val  = max(1.0, float(data.get('INR', 1.1)))
        creat_val = max(1.0, creat)

        meld_score = round(3.78 * math.log(bili_val) + 11.2 * math.log(inr_val) + 9.57 * math.log(creat_val) + 6.43)
        if meld_score < 10:
            meld_severity = "Low severity"
        elif meld_score <= 19:
            meld_severity = "Moderate severity"
        elif meld_score <= 29:
            meld_severity = "High severity"
        else:
            meld_severity = "Critical"

        # Compare MELD with AI Risk dynamically
        if risk_level == "High" and meld_score >= 20:
            meld_comparison = "Strong Agreement: Both AI Risk and MELD score indicate severe clinical decompensation and high mortality risk."
        elif risk_level == "Moderate" and 10 <= meld_score <= 19:
            meld_comparison = "Agreement: Both models indicate moderate clinical severity and early decompensation."
        elif risk_level == "Low" and meld_score < 10:
            meld_comparison = "Agreement: Both models indicate low severity with compensated liver function."
        elif risk_level == "High" and meld_score < 20:
            meld_comparison = "Divergence (AI Higher): AI Risk is significantly higher than MELD. This is likely driven by extreme deviations in systemic markers (like Sodium or WBC) which MELD does not track."
        elif risk_level in ["Low", "Moderate"] and meld_score >= 20:
            meld_comparison = "Divergence (MELD Higher): MELD score is significantly higher than AI Risk. This suggests severe biliary or coagulation issues (high Bilirubin/INR) without significant electrolyte or inflammatory crisis."
        else:
            meld_comparison = "Minor Divergence: Models show slightly differing risk categories, but overall clinical trajectory requires close monitoring."

        # ── Doctor notes (formal) ─────────────────────────────────────────
        doctor_notes = (
            f'Patient {patient_name} (ID: {patient_id}), {int(age)}-year-old '
            f'{("male" if gender=="M" else "female")}, presents with a {risk_level.lower()} '
            f'cirrhosis severity risk ({prob:.1f}%) and MELD score {meld_score} ({meld_severity}). '
            f'Key lab abnormalities: {"; ".join([f.split("—")[0].strip() for f in findings[:3]])}. '
            f'Clinical urgency: {urgency}. '
            f'Immediate action: {actions[0]} {actions[1] if len(actions) > 1 else ""}'
        )

        copy_summary = f"{risk_level} mortality risk with MELD score {meld_score} indicating {meld_severity.lower()}. "
        if creat >= 1.5 or sodium < 135:
            copy_summary += "Renal function and electrolyte imbalance require close monitoring."
        else:
            copy_summary += "Current liver and renal markers should be monitored routinely."

        return jsonify({
            'status': 'success',
            'insight': {
                'risk_level':        risk_level,
                'risk_color':        risk_color,
                'risk_percentage':   round(prob, 1),
                'risk_interpretation': risk_interp,
                'meld_score':        meld_score,
                'meld_severity':     meld_severity,
                'meld_comparison':   meld_comparison,
                'copy_summary':      copy_summary,
                'findings':          findings,
                'complications':     complications,
                'actions':           actions,
                'urgency':           urgency,
                'urgency_reason':    urgency_reason,
                'confidence':        confidence,
                'trend':             trend,
                'trend_interpretation': trend_interp,
                'trend_direction':   trend_dir,
                'alerts':            alerts,
                'abnormal_labs':     abnormal_labs,
                'contributions':     contribution_list,
                'differential':      differential,
                'quick_summary':     quick_summary,
                'whatif':            whatif,
            },
            'report': {
                'patient_id':    patient_id,
                'patient_name':  patient_name,
                'age':           int(age),
                'gender':        'Male' if gender == 'M' else 'Female',
                'lab': {
                    'sodium':     {'value': sodium, 'status': lab_status['sodium'][0],  'severity': lab_status['sodium'][1]},
                    'creatinine': {'value': creat,  'status': lab_status['creat'][0],   'severity': lab_status['creat'][1]},
                    'urea':       {'value': urea,   'status': lab_status['urea'][0],    'severity': lab_status['urea'][1]},
                    'hemoglobin': {'value': hemo,   'status': lab_status['hemo'][0],    'severity': lab_status['hemo'][1]},
                    'wbc':        {'value': wbc,    'status': lab_status['wbc'][0],     'severity': lab_status['wbc'][1]},
                },
                'risk_percentage':    round(prob, 1),
                'risk_level':         risk_level,
                'risk_interpretation': risk_interp,
                'meld_score':         meld_score,
                'meld_severity':      meld_severity,
                'copy_summary':       copy_summary,
                'findings':           findings,
                'complications':      complications,
                'actions':            actions,
                'differential':       differential,
                'quick_summary':      quick_summary,
                'whatif':             whatif,
                'trend':              trend,
                'trend_interpretation': trend_interp,
                'doctor_notes':       doctor_notes,
                'generated_at':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == '__main__':
    app.run(port=5000, debug=True)
