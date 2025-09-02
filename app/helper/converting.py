

# ✅ الحدود الطبيعية لكل التحاليل
NORMAL_RANGES = {
    "GLUCOSE": (70, 110),             # mg/dL
    "UREA": (15, 45),                 # mg/dL
    "CREATININE": (0.6, 1.3),         # mg/dL
    "URIC_ACID": (3.5, 7.2),          # mg/dL
    "EGFR": (90, 120),                # mL/min/1.73m2
    "CALCIUM": (8.5, 10.5),           # mg/dL
    "MAGNESIUM": (1.7, 2.2),          # mg/dL
    "SODIUM": (135, 145),             # mmol/L
    "POTASSIUM": (3.5, 5.1),          # mmol/L
    "CHLORIDE": (98, 107),            # mmol/L
    "TRIGLYCERIDES": (0, 150),        # mg/dL
    "LDL": (0, 100),                   # mg/dL
    "HDL": (40, 60),                   # mg/dL
    "TOTAL_CHOLESTEROL": (125, 200),   # mg/dL
    "VLDL": (5, 40),                    # mg/dL
    "ALT": (7, 56),                     # U/L
    "AST": (10, 40),                    # U/L
    "ALP": (44, 147),                   # U/L
    "GGT": (9, 48),                     # U/L
    "BILIRUBIN_TOTAL": (0.1, 1.2),      # mg/dL
    "BILIRUBIN_DIRECT": (0, 0.3),       # mg/dL
    "CBC": None,                        # يشمل العديد من القيم، كل واحدة لها نطاق
    "CRP": (0, 5),                       # mg/L
    "ESR": (0, 20),                       # mm/hr
    "ANA": None,                          # qualitative
    "RF": (0, 14),                        # IU/mL
    "PROCALCITONIN": (0, 0.1),            # ng/mL
    "IGG": (700, 1600),                    # mg/dL
    "IGM": (40, 230),                      # mg/dL
    "IGA": (70, 400),                      # mg/dL
    "TSH": (0.4, 4.0),                     # µIU/mL
    "TESTOSTERONE": (300, 1000),           # ng/dL (رجال)
    "ESTROGEN": (15, 350),                 # pg/mL
    "PROGESTERONE": (0.1, 25),             # ng/mL
    "LH": (1.8, 8.6),                       # IU/L
    "FSH": (1.5, 12.4),                     # IU/L
    "PROLACTIN": (4.8, 23.3),               # ng/mL
    "INSULIN": (2, 25),                      # µIU/mL
    "VITAMIN_D": (30, 100),                 # ng/mL
    "VITAMIN_B12": (200, 900),              # pg/mL
    "FOLATE": (2.7, 17),                    # ng/mL
    "URINE_ANALYSIS": None,                  # qualitative
    "PSA": (0, 4),                           # ng/mL
    "CA_125": (0, 35),                        # U/mL
    "CA19_9": (0, 37),                        # U/mL
    "CEA": (0, 3),                            # ng/mL
    "AFP": (0, 10),                           # ng/mL
    "CK": (20, 200),                           # U/L
    "TROPONIN": (0, 0.04),                     # ng/mL
    "LDH": (140, 280),                          # U/L
    "MYOGLOBIN": (28, 72),                      # ng/mL
    "BETA_HCG": (0, 5),                         # mIU/mL
    "CO2": (23, 29),                             # mmol/L
    "ANION_GAP": (8, 16),                        # mmol/L
    "OSMOLARITY": (275, 295),                     # mOsm/kg
    "LACTATE": (0.5, 2.2),                        # mmol/L
    "AMYLASE": (23, 85),                           # U/L
    "LIPASE": (0, 160)                             # U/L
}

# ✅ أسماء بالعربي لكل التحاليل
TEST_NAMES_AR = {
    "GLUCOSE": "غلوكوز",
    "UREA": "يوريا",
    "CREATININE": "كرياتينين",
    "URIC_ACID": "حمض اليوريك",
    "EGFR": "معدل الترشيح الكبيبي",
    "CALCIUM": "كالسيوم",
    "MAGNESIUM": "مغنيسيوم",
    "SODIUM": "صوديوم",
    "POTASSIUM": "بوتاسيوم",
    "CHLORIDE": "كلوريد",
    "TRIGLYCERIDES": "دهون ثلاثية",
    "LDL": "كوليسترول ضار (LDL)",
    "HDL": "كوليسترول نافع (HDL)",
    "TOTAL_CHOLESTEROL": "كوليسترول كلي",
    "VLDL": "كوليسترول VLDL",
    "ALT": "ALT (SGPT)",
    "AST": "AST (SGOT)",
    "ALP": "ALP",
    "GGT": "GGT",
    "BILIRUBIN_TOTAL": "بيليروبين كلي",
    "BILIRUBIN_DIRECT": "بيليروبين مباشر",
    "CBC": "تحليل دم كامل",
    "CRP": "CRP",
    "ESR": "ESR",
    "ANA": "ANA",
    "RF": "RF",
    "PROCALCITONIN": "Procalcitonin",
    "IGG": "IgG",
    "IGM": "IgM",
    "IGA": "IgA",
    "TSH": "TSH و الغدة الدرقية",
    "TESTOSTERONE": "تستوستيرون",
    "ESTROGEN": "إستروجين",
    "PROGESTERONE": "بروجستيرون",
    "LH": "LH",
    "FSH": "FSH",
    "PROLACTIN": "برولاكتين",
    "INSULIN": "إنسولين",
    "VITAMIN_D": "فيتامين د",
    "VITAMIN_B12": "فيتامين ب12",
    "FOLATE": "فولات",
    "URINE_ANALYSIS": "تحليل بول",
    "PSA": "PSA",
    "CA_125": "CA-125",
    "CA19_9": "CA19-9",
    "CEA": "CEA",
    "AFP": "AFP",
    "CK": "CK",
    "TROPONIN": "Troponin",
    "LDH": "LDH",
    "MYOGLOBIN": "Myoglobin",
    "BETA_HCG": "Beta HCG",
    "CO2": "CO2",
    "ANION_GAP": "Anion Gap",
    "OSMOLARITY": "Osmolarity",
    "LACTATE": "Lactate",
    "AMYLASE": "Amylase",
    "LIPASE": "Lipase"
}

def analyze_results(results: dict):
    report = {}
    for test, value in results.items():
        try:
            val = float(value)
            display_name = TEST_NAMES_AR.get(test, test)  # لو مش موجود في القاموس استعمل نفس الاسم
            if test in NORMAL_RANGES:
                low, high = NORMAL_RANGES[test]
                if val < low:
                    report[display_name] = {
                        "القيمة": val,
                        "الحالة": f"↓ أقل من الطبيعي",
                        "المدى الطبيعي": f"{low}-{high}"
                    }
                elif val > high:
                    report[display_name] = {
                        "القيمة": val,
                        "الحالة": f"↑ أعلى من الطبيعي",
                        "المدى الطبيعي": f"{low}-{high}"
                    }
                else:
                    report[display_name] = {
                        "القيمة": val,
                        "الحالة": "✅ ضمن الطبيعي",
                        "المدى الطبيعي": f"{low}-{high}"
                    }
            else:
                report[display_name] = {
                    "القيمة": val,
                    "الحالة": "لا يوجد مدى طبيعي مسجل"
                }
        except ValueError:
            report[test] = {
                "القيمة": value,
                "الحالة": "قيمة غير رقمية"
            }
    return report
