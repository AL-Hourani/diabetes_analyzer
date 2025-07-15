

def generate_report(analyzed_results, interpretations):
    report_lines = []

    for test, status in analyzed_results.items():
       
        if status == "غير مدعوم":
            continue

        test_key = test.lower()
        if test_key in interpretations and status in interpretations[test_key]:
            info = interpretations[test_key]
            name = info["name"]
            explanation = info[status]
            line = f"قيمة {name} لديك {status} {explanation}"
            report_lines.append(line)

    return " ".join(report_lines)
