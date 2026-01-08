# analysis/risk_analysis.py

def calculate_risk(fire_detected=False, crowd_count=0, traffic_level=0):
    """
    Simple risk score calculator
    """

    risk_score = 0

    if fire_detected:
        risk_score += 70

    if crowd_count > 20:
        risk_score += 20
    elif crowd_count > 10:
        risk_score += 10

    if traffic_level > 15:
        risk_score += 10

    if risk_score >= 80:
        level = "HIGH RISK 🔴"
    elif risk_score >= 40:
        level = "MEDIUM RISK 🟠"
    else:
        level = "LOW RISK 🟢"

    return risk_score, level
