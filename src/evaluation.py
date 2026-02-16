import re
import pandas as pd
from workflow import app, prices, calc_total_return, calc_volatility

# -----------------------------------
# Test Set
# -----------------------------------
TEST_CASES = [
    {
        "query": "What is the total return of AAPL?",
        "type": "numeric",
        "ticker": "AAPL",
        "metric": "return",
    },
    {
        "query": "Compute TSLA volatility",
        "type": "numeric",
        "ticker": "TSLA",
        "metric": "volatility",
    },
    {
        "query": "Summarize the recent trend of MSFT",
        "type": "rag",
    },
    {
        "query": "What happened on 2024-01-05 for AAPL?",
        "type": "rag",
    },
]

results = []

print("\n===== Running Evaluation =====")

for case in TEST_CASES:
    query = case["query"]
    expected_type = case["type"]

    out = app.invoke({"user_query": query})

    answer = out["answer"]
    route = out["route"]
    confidence = out["confidence"]
    hallucination = out.get("hallucination_risk", False)

    record = {
        "query": query,
        "expected_type": expected_type,
        "route": route,
        "routing_correct": route == expected_type,
        "confidence": confidence,
        "hallucination_risk": hallucination,
    }

    # -----------------------
    # Numeric correctness
    # -----------------------
    if expected_type == "numeric":
        ticker = case["ticker"]

        if case["metric"] == "return":
            truth = calc_total_return(prices, ticker)
        else:
            truth = calc_volatility(prices, ticker)

        # Extract number from answer
        match = re.search(r"([-+]?\d+\.\d+)", answer)
        if match:
            predicted = float(match.group(1)) / 100
            error = abs(predicted - truth)
            record["numeric_error"] = error
            record["numeric_correct"] = error < 0.01
        else:
            record["numeric_error"] = None
            record["numeric_correct"] = False

    results.append(record)

df = pd.DataFrame(results)

print("\n===== Evaluation Summary =====")
print(df[[
    "query",
    "routing_correct",
    "confidence",
    "hallucination_risk"
]])

routing_acc = df["routing_correct"].mean()
print("\nRouting Accuracy:", routing_acc)

if "numeric_correct" in df.columns:
    numeric_acc = df["numeric_correct"].mean()
    print("Numeric Accuracy:", numeric_acc)

if "hallucination_risk" in df.columns:
    halluc_rate = df["hallucination_risk"].mean()
    print("Hallucination Risk Rate:", halluc_rate)

df.to_csv("evaluation_results.csv", index=False)
print("\nSaved evaluation_results.csv")
