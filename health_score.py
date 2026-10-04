
def get_numeric_value(nutrition, key):
    """Extract a numeric nutrition value safely."""

    item = nutrition.get(key)

    if not isinstance(item, dict):
        return None

    value = item.get("value")

    if value is None:
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def score_lower_is_better(value, excellent, moderate, high):
    """Score nutrients where lower values are preferred."""

    if value is None:
        return None

    if value <= excellent:
        return 100

    if value <= moderate:
        return 70

    if value <= high:
        return 40

    return 10


def score_higher_is_better(value, low, good):
    """Score nutrients where higher values are preferred."""

    if value is None:
        return None

    if value >= good:
        return 100

    if value >= low:
        return 70

    return 30


def calculate_health_score(analysis):

    if not isinstance(analysis, dict):
        return None

    nutrition = analysis.get("nutrition", {})

    if not isinstance(nutrition, dict):
        return None

    factors = {
        "calories": {
            "weight": 15,
            "score": score_lower_is_better(
                get_numeric_value(nutrition, "calories"),
                100, 250, 400
            )
        },

        "total_sugars": {
            "weight": 20,
            "score": score_lower_is_better(
                get_numeric_value(nutrition, "total_sugars"),
                5, 10, 20
            )
        },

        "saturated_fat": {
            "weight": 15,
            "score": score_lower_is_better(
                get_numeric_value(nutrition, "saturated_fat"),
                1.5, 3, 5
            )
        },

        "sodium": {
            "weight": 15,
            "score": score_lower_is_better(
                get_numeric_value(nutrition, "sodium"),
                120, 400, 600
            )
        },

        "protein": {
            "weight": 15,
            "score": score_higher_is_better(
                get_numeric_value(nutrition, "protein"),
                5, 10
            )
        },

        "fiber": {
            "weight": 10,
            "score": score_higher_is_better(
                get_numeric_value(nutrition, "fiber"),
                3, 6
            )
        },

        "added_sugars": {
            "weight": 10,
            "score": score_lower_is_better(
                get_numeric_value(nutrition, "added_sugars"),
                2, 5, 10
            )
        }
    }

    available = [
        item for item in factors.values()
        if item["score"] is not None
    ]

    if len(available) < 3:
        return {
            "score": None,
            "category": "Insufficient data",
            "available_factors": len(available),
            "total_factors": len(factors)
        }

    total_weight = sum(
        item["weight"] for item in available
    )

    weighted_score = sum(
        item["score"] * item["weight"]
        for item in available
    )

    final_score = round(weighted_score / total_weight)

    if final_score >= 80:
        category = "Better nutritional profile"
    elif final_score >= 60:
        category = "Moderate nutritional profile"
    elif final_score >= 40:
        category = "Consider limiting"
    else:
        category = "Less favorable nutritional profile"

    return {
        "score": final_score,
        "category": category,
        "available_factors": len(available),
        "total_factors": len(factors)
    }
