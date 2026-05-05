import urllib.request, json

def test(label, age, gender, creat, hemo, sodium, urea, wbc):
    payload = json.dumps({
        'anchor_age': age, 'gender': gender,
        'Creatinine': creat, 'Hemoglobin': hemo,
        'Sodium': sodium, 'Urea Nitrogen': urea, 'WBC': wbc
    }).encode()
    req = urllib.request.Request(
        'http://localhost:5000/predict', method='POST',
        headers={'Content-Type': 'application/json'}, data=payload
    )
    res = json.loads(urllib.request.urlopen(req).read())
    print(label, "=>", res["mortality_probability"], "%")

test("Moderate (Male)  ", 65, "M", 1.5, 9.5, 133, 60, 9)
test("Moderate (Female)", 65, "F", 1.5, 9.5, 133, 60, 9)
test("Low risk labs    ", 45, "M", 1.0, 13.0, 140, 15, 7)
test("High risk labs   ", 65, "M", 2.0, 8.0, 126, 60, 14)
