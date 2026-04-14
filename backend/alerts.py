import cv2

def get_alerts(v_score, lines):
    """Visibility aur Lanes ke basis par alerts dena."""
    alerts = []
    
    # Thresholds: 40% se kam visibility matlab khatra
    if v_score < 40:
        alerts.append(("CRITICAL: LOW VISIBILITY", (0, 0, 255))) # Red Alert
    elif v_score < 60:
        alerts.append(("CAUTION: MODERATE FOG", (0, 255, 255))) # Yellow Warning

    # Lane Alert
    if lines is None or len(lines) == 0:
        alerts.append(("WARNING: LANE LOST", (0, 165, 255))) # Orange Alert
        
    return alerts