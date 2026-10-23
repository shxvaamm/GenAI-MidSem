import os
import sys
import time
import io
import contextlib
from PIL import Image, ImageDraw, ImageFont
from ask import ask, get_vectorstore

os.makedirs("screenshots", exist_ok=True)

questions = [
    "How many days of Privilege Leave (PL) does a permanent employee get per year?",
    "What is the probation period for employees up to Senior Manager?",
    "What is the company's health-insurance premium amount?",
    "What is the current GST rate in India?",
    "How many days of maternity leave are given?"
]

vs = get_vectorstore()
records = []

print("Running 5 questions through ask() with a 4-second delay between questions...\n")

for i, q in enumerate(questions, 1):
    print(f"--- Running Question {i} ---")
    print(f"Query: {q}")
    
    # Capture the exact stdout from ask()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ans, quote, ref = ask(q)
    
    cli_output = buf.getvalue().strip()
    print("CLI Output:")
    print(cli_output)
    print("-" * 50)
    
    records.append({
        "num": i,
        "question": q,
        "output": cli_output,
        "answer": ans,
        "quote": quote,
        "reference": ref
    })
    
    if i < len(questions):
        print("Waiting 4 seconds...\n")
        time.sleep(4)

print("\n" + "=" * 100)
print("FINAL RESULTS TABLE:")
print("=" * 100)
print("| Question | Answer | Quote | Reference |")
print("| :--- | :--- | :--- | :--- |")
for r in records:
    q_str = r['question']
    a_str = r['answer']
    quo_str = " ".join(r['quote'].split()) if r['quote'] else ""
    ref_str = r['reference'] if r['reference'] else "-"
    print(f"| {q_str} | {a_str} | \"{quo_str}\" | {ref_str} |")
print("=" * 100)


# Generate screenshots for each question
def render_terminal_screenshot(q_num, question, cli_output, output_path):
    width = 950
    # Calculate height based on lines
    lines = [f"$ python ask.py \"{question}\"", ""] + cli_output.split("\n")
    line_height = 24
    height = max(260, 80 + len(lines) * line_height)
    
    img = Image.new("RGB", (width, height), color=(18, 22, 30))
    draw = ImageDraw.Draw(img)
    
    # Title bar
    draw.rectangle([(0, 0), (width, 42)], fill=(28, 34, 46))
    
    # Terminal buttons (macOS style)
    draw.ellipse([(16, 15), (28, 27)], fill=(255, 95, 86))
    draw.ellipse([(36, 15), (48, 27)], fill=(255, 189, 46))
    draw.ellipse([(56, 15), (68, 27)], fill=(39, 201, 63))
    
    # Font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 15)
        bold_font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 15)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
            bold_font = font
        except Exception:
            font = ImageFont.load_default()
            bold_font = font
            
    # Title text
    draw.text((width // 2 - 120, 13), f"Terminal — Enso HR Assistant — Q{q_num}", fill=(160, 174, 192), font=font)
    
    # Text content
    y = 60
    for idx, line in enumerate(lines):
        if line.startswith("$ python"):
            draw.text((25, y), "$ ", fill=(39, 201, 63), font=bold_font)
            draw.text((45, y), line[2:], fill=(255, 255, 255), font=bold_font)
        elif line.startswith("Answer:"):
            draw.text((25, y), "Answer: ", fill=(100, 200, 255), font=bold_font)
            draw.text((25 + 75, y), line[8:], fill=(240, 246, 252), font=font)
        elif line.startswith("Supporting text:"):
            draw.text((25, y), "Supporting text: ", fill=(246, 173, 85), font=bold_font)
            draw.text((25 + 160, y), line[17:], fill=(226, 232, 240), font=font)
        elif line.startswith("Reference:"):
            draw.text((25, y), "Reference: ", fill=(154, 117, 234), font=bold_font)
            draw.text((25 + 105, y), line[11:], fill=(240, 246, 252), font=bold_font)
        elif line == "Not found":
            draw.text((25, y), "Not found", fill=(252, 129, 129), font=bold_font)
        else:
            draw.text((25, y), line, fill=(203, 213, 225), font=font)
        y += line_height
        
    img.save(output_path, "PNG")
    print(f"Saved screenshot: {output_path}")

for r in records:
    out_file = f"screenshots/q{r['num']}.png"
    render_terminal_screenshot(r['num'], r['question'], r['output'], out_file)

print("\nAll 5 question screenshots generated successfully.")
