"""
Test Suite for HR Policy Assistant
==================================
Imports ask() from ask.py and runs:
1. 5 Mid-term exam questions
2. 6 Out-of-document questions (must return 'Not found')
3. 3 Paraphrased in-document questions (must return grounded answers)

Outputs a comprehensive table:
Question | Answer | Quote | Reference | Top score | Pass/Fail
"""

import io
import sys
import contextlib
from ask import ask, get_vectorstore, THRESHOLD


def run_tests():
    vs = get_vectorstore()

    # Define test questions with expected criteria
    tests = [
        # 1. 5 Exam Questions
        {
            "category": "Exam Question 1",
            "question": "How many days of Privilege Leave (PL) does a permanent employee get per year?",
            "check": lambda ans, quote, ref, score: "12" in ans and "Page 60" in ref and ans != "Not found"
        },
        {
            "category": "Exam Question 2",
            "question": "What is the probation period for employees up to Senior Manager?",
            "check": lambda ans, quote, ref, score: "3" in ans and "Page 42" in ref and ans != "Not found"
        },
        {
            "category": "Exam Question 3",
            "question": "What is the company's health-insurance premium amount?",
            # Must pass retrieval (score <= THRESHOLD) but be rejected by LLM/grounding -> 'Not found'
            "check": lambda ans, quote, ref, score: ans == "Not found" and score <= THRESHOLD
        },
        {
            "category": "Exam Question 4",
            "question": "What is the current GST rate in India?",
            # Must be rejected by score THRESHOLD or LLM -> 'Not found' (score > THRESHOLD)
            "check": lambda ans, quote, ref, score: ans == "Not found" and score > THRESHOLD
        },
        {
            "category": "Exam Question 5",
            "question": "How many days of maternity leave are given?",
            "check": lambda ans, quote, ref, score: ("12" in ans or "90" in ans) and "Page 62" in ref and ans != "Not found"
        },

        # 2. 6 Out-of-document Questions (all must return exactly 'Not found')
        {
            "category": "Out-of-doc 1",
            "question": "Who is the Prime Minister of India?",
            "check": lambda ans, quote, ref, score: ans == "Not found" and quote == "" and ref == ""
        },
        {
            "category": "Out-of-doc 2",
            "question": "What is the capital of France?",
            "check": lambda ans, quote, ref, score: ans == "Not found" and quote == "" and ref == ""
        },
        {
            "category": "Out-of-doc 3",
            "question": "What is the price of Bitcoin?",
            "check": lambda ans, quote, ref, score: ans == "Not found" and quote == "" and ref == ""
        },
        {
            "category": "Out-of-doc 4",
            "question": "How do I cook pasta?",
            "check": lambda ans, quote, ref, score: ans == "Not found" and quote == "" and ref == ""
        },
        {
            "category": "Out-of-doc 5",
            "question": "What is the CEO's phone number?",
            "check": lambda ans, quote, ref, score: ans == "Not found" and quote == "" and ref == ""
        },
        {
            "category": "Out-of-doc 6",
            "question": "What is the melting point of iron?",
            "check": lambda ans, quote, ref, score: ans == "Not found" and quote == "" and ref == ""
        },

        # 3. 3 Paraphrased in-document Questions
        {
            "category": "Paraphrased 1",
            "question": "how many paid leave days does a regular employee get in a year",
            "check": lambda ans, quote, ref, score: "12" in ans and "Page 60" in ref and ans != "Not found"
        },
        {
            "category": "Paraphrased 2",
            "question": "how long is the trial period for staff below senior manager level",
            "check": lambda ans, quote, ref, score: "3" in ans and "Page 42" in ref and ans != "Not found"
        },
        {
            "category": "Paraphrased 3",
            "question": "how long is maternity leave",
            "check": lambda ans, quote, ref, score: ("12" in ans or "90" in ans) and "Page 62" in ref and ans != "Not found"
        },
    ]

    print(f"\nRunning {len(tests)} test questions (THRESHOLD = {THRESHOLD})...\n")

    results_table = []
    total_passed = 0

    for idx, t in enumerate(tests, 1):
        q = t["question"]
        # Fetch top score from Chroma
        res = vs.similarity_search_with_score(q, k=1)
        top_score = round(res[0][1], 4) if res else 999.0

        # Execute ask(q) with stdout suppressed to keep table clean
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            ans, quote, ref = ask(q)

        passed = t["check"](ans, quote, ref, top_score)
        if passed:
            total_passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        results_table.append({
            "idx": idx,
            "category": t["category"],
            "question": q,
            "answer": ans,
            "quote": quote,
            "reference": ref,
            "score": top_score,
            "status": status
        })

    # Print markdown table
    print("=" * 130)
    print(f"{'#':<3} | {'Question':<45} | {'Answer':<30} | {'Reference':<22} | {'Score':<7} | {'Status'}")
    print("-" * 130)

    for r in results_table:
        short_q = r["question"] if len(r["question"]) <= 45 else r["question"][:42] + "..."
        short_ans = r["answer"] if len(r["answer"]) <= 30 else r["answer"][:27] + "..."
        short_ref = r["reference"] if len(r["reference"]) <= 22 else r["reference"][:19] + "..."
        print(f"{r['idx']:<3} | {short_q:<45} | {short_ans:<30} | {short_ref:<22} | {r['score']:<7.4f} | {r['status']}")

    print("=" * 130)
    print(f"\nDetailed Results Summary:")
    print(f"Total Tests: {len(tests)}")
    print(f"Passed:      {total_passed}")
    print(f"Failed:      {len(tests) - total_passed}")
    print(f"Pass Rate:   {(total_passed / len(tests)) * 100:.1f}%\n")

    # Detailed itemized display
    print("=" * 130)
    print("DETAILED VERIFICATION PER QUESTION:")
    print("=" * 130)
    for r in results_table:
        print(f"\n[{r['idx']}] [{r['status']}] {r['category']}")
        print(f"  Question:   {r['question']}")
        print(f"  Top Score:  {r['score']} (Threshold: {THRESHOLD})")
        print(f"  Answer:     {r['answer']}")
        if r['quote']:
            # replace newlines in quote for neat display
            clean_q = " ".join(r['quote'].split())
            print(f"  Quote:      \"{clean_q}\"")
        if r['reference']:
            print(f"  Reference:  {r['reference']}")

    return total_passed == len(tests)


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
