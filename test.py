from metrics import compute_metric

print(
    compute_metric(
        "bertscore", "I have a cock a.", "I have a cock bock of shit and suck."
    )
)
