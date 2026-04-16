import cv2

def get_alerts(v_score, lines):
    """Generate real-time driver safety alerts based on visibility score and lane detection status."""
    alerts = []

    # Visibility threshold logic — critical below 40%, caution between 40% and 60%
    if v_score < 40:
        alerts.append(("CRITICAL: LOW VISIBILITY", (0, 0, 255)))    # Red — immediate danger
    elif v_score < 60:
        alerts.append(("CAUTION: MODERATE FOG", (0, 255, 255)))     # Yellow — reduced safety margin

    # Lane alert — trigger if no geometric lane lines were detected by the Hough Transform
    if lines is None or len(lines) == 0:
        alerts.append(("WARNING: LANE LOST", (0, 165, 255)))        # Orange — lane tracking failure

    return alerts