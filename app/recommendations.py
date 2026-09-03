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


# Returned instead of a class recommendation when the reading falls outside the
# domain the model was fitted on. The class label is still reported, but it is
# an extrapolation and must not drive an automated action.
OUT_OF_DOMAIN = {
    "title": "Reading Outside Validated Range",
    "message": (
        "One or more inputs are far outside the range this model was trained "
        "on. The classification below is an extrapolation, not a measurement."
    ),
    "actions": [
        "Verify the sensor is connected and reading correctly",
        "Confirm the rate values are 60-second least-squares slopes",
        "Inspect the cargo manually rather than relying on this result",
    ],
    "next_check": "Immediately",
}


def get_recommendation(class_id, in_domain=True):
    if not in_domain:
        return OUT_OF_DOMAIN

    return RECOMMENDATIONS[class_id]