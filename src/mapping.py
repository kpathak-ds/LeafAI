def map_label_to_efficiency(label):
    label = label.lower()

    if "healthy" in label:
        return 0.9
    
    elif "drought" in label:
        return 0.6   # water stress
    
    elif "nutrient" in label:
        return 0.65  # nitrogen issue
    
    elif "stress" in label:
        return 0.6

    return 0.7