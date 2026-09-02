RECOMMENDATIONS = {
    0: {
        "title": "All Clear",
        "message": "All readings within safe range",
        "actions": [
            "No action needed"
        ],
        "next_check": "2 hours"
    },

    1: {
        "title": "Warning",
        "message": "Environmental conditions require attention",
        "actions": [
            "Check temperature and humidity",
            "Increase ventilation if appropriate",
            "Inspect the storage or transport environment"
        ],
        "next_check": "1 hour"
    },

    2: {
        "title": "Action Required",
        "message": "Immediate intervention is recommended",
        "actions": [
            "Stop transport immediately",
            "Inspect for spoilage now",
            "Alert destination facility"
        ],
        "next_check": "Immediately"
    }
}


def get_recommendation(class_id):
    return RECOMMENDATIONS[class_id]