"""HAM10000 diagnosis codes and their integer labels."""

# Order is fixed so that label ids stay stable across runs and checkpoints.
CLASSES: tuple[str, ...] = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")

CLASS_NAMES: dict[str, str] = {
    "akiec": "Actinic keratoses / Bowen's disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
