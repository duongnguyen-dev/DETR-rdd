CLASS = {
    "D00": {
        "name": "longitudinal_crack",
        "color": "red"
    },
    "D10": {
        "name": "transverse_crack",
        "color": "green"
    },
    "D20": {
        "name": "aligator_crack",
        "color": "yellow"
    },
    "D40": {
        "name": "pothole",
        "color": "blue"
    },
}

CLASS_MAPPING = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "aligator_crack",
    3: "pothole",
    4: "other_corruptions"
}

CHECKPOINT = "facebook/detr-resnet-50"

OUTPUT_DIR = ""

COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]
