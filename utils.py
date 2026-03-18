import math
from collections.abc import Mapping, Sequence


def validate_data(data):
    if data is None:
        return False
    if isinstance(data, (str, bytes)):
        return True
    if isinstance(data, (int, bool)):
        return True
    if isinstance(data, float):
        return not (math.isnan(data) or math.isinf(data))
    if isinstance(data, Mapping):
        return all(validate_data(value) for value in data.values())
    if isinstance(data, Sequence):
        return all(validate_data(value) for value in data)
    return True


def format_data(data, decimals=4):
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return round(data, decimals)
    if isinstance(data, Mapping):
        return {key: format_data(value, decimals=decimals) for key, value in data.items()}
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return [format_data(value, decimals=decimals) for value in data]
    return data


def calculate(data):
    values = []

    def collect(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if not math.isnan(number) and not math.isinf(number):
                values.append(number)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                collect(item)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                collect(item)

    collect(data)

    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "sum": 0.0}

    total = sum(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": total / len(values),
        "sum": total,
    }
