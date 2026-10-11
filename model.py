def classify_emotion(left_mean, right_mean, variance, file_tag):
    """
    Classify a sensor-data row using the existing heuristic rules.
    Returns the valence score and mapped emotion label.
    """

    valence_score = left_mean - right_mean
    file_tag = str(file_tag).strip().upper()

    if file_tag == "NEUTRAL":
        if variance < 15:
            mapped_emotion = "DROWSY"
        else:
            mapped_emotion = "RELAXED"

    elif file_tag == "POSITIVE":
        if variance > 30:
            mapped_emotion = "HAPPY"
        else:
            mapped_emotion = "CONTENT"

    elif file_tag == "NEGATIVE":
        if variance > 50:
            mapped_emotion = "ANGRY"
        elif valence_score < 0:
            mapped_emotion = "SAD"
        else:
            mapped_emotion = "DISTRESSED"

    else:
        mapped_emotion = "UNDETERMINED"

    return valence_score, mapped_emotion