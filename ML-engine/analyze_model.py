import joblib, pandas as pd

m = joblib.load('xgb_mortality_model.pkl')
cols = joblib.load('xgb_feature_columns.pkl')

results = []
for age in [40, 55, 65, 75]:
    for g in ['F','M']:
        for creat in [1.3, 1.5, 2.0]:
            for sodium in [130, 133, 136]:
                for hemo in [9.5, 10.5]:
                    for urea in [25, 40, 60]:
                        for wbc in [5, 9, 14]:
                            bun = urea/creat
                            row = {'anchor_age':age,'gender':1 if g=='M' else 0,'Creatinine':creat,'Hemoglobin':hemo,
                                   'Sodium':sodium,'Urea Nitrogen':urea,'WBC':wbc,'bun_creatinine_ratio':bun,
                                   'renal_dysfunction_flag':int(creat>=1.5),'severe_anemia':int(hemo<9),
                                   'hyponatremia':int(sodium<135),'severe_hyponatremia':int(sodium<130),
                                   'leukocytosis':int(wbc>11),'elderly_risk':int(age>=60)}
                            p = m.predict_proba(pd.DataFrame([row])[cols])[0][1]*100
                            results.append((round(p,1), age, g, creat, hemo, sodium, urea, wbc))

results.sort()
probs = [r[0] for r in results]
buckets = {'<10':0,'10-25':0,'25-50':0,'50-75':0,'>75':0}
for p in probs:
    if p < 10: buckets['<10']+=1
    elif p < 25: buckets['10-25']+=1
    elif p < 50: buckets['25-50']+=1
    elif p < 75: buckets['50-75']+=1
    else: buckets['>75']+=1

print('=== Model Output Distribution ===')
for k,v in buckets.items(): print(f'  {k}%: {v} cases')
print(f'Min: {min(probs)}   Max: {max(probs)}')

print('\n=== Moderate zone (25-70%) combos ===')
for r in [x for x in results if 25 <= x[0] <= 70]:
    print(f'{r[0]:.1f}% | age={r[1]} g={r[2]} creat={r[3]} Na={r[5]} urea={r[6]} hemo={r[4]} wbc={r[7]}')
