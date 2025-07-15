

from rapidfuzz import process, fuzz

MEDICAL_KEYWORDS = [
   
    
    "glucose", "glu", "sugar", "fbs", "rbs", "hba1c","calcium","total cholesterol",

    "cholesterol", "ldl", "hdl", "vldl", "triglycerides", "tg", "lipid", "lipoprotein",

  
    "urea", "bun", "creatinine", "uric acid", "egfr","ua"

   
    "alt", "sgpt", "ast", "sgot", "alp", "ggt", "bilirubin", "total bilirubin", "direct bilirubin",

    "sodium", "na", "potassium", "k", "chloride", "cl", "calcium", "ca", "phosphate", "mg", "magnesium",

 
    "cbc", "wbc", "rbc", "hgb", "hemoglobin", "hct", "hematocrit", "mcv", "mch", "mchc", "plt", "platelets",


    "crp", "hs-crp", "esr", "ana", "rf", "procalcitonin", "igg", "igm", "iga",

    "tsh", "ft3", "ft4", "t3", "t4", "testosterone", "estrogen", "progesterone", "lh", "fsh", "prolactin", "insulin",

    "tsh", "t3", "t4", "anti-tpo", "anti-tg",

  
    "vitamin d", "vit d", "vitamin b12", "folate", "folic acid",

    "hba1c", "fbs", "rbs", "ppbs",

  
    "urine", "stool", "albumin", "protein", "ketone", "bilirubin", "urobilinogen", "ph", "specific gravity", "pus cells", "rbc in urine",

    "psa", "ca-125", "ca19-9", "cea", "afp",

  
    "ck", "ck-mb", "troponin", "ldh", "myoglobin",

    
    "beta hcg", "hcg",

 
    "co2", "anion gap", "osmolarity", "lactate", "amylase", "lipase" , "ua","tc","tc m",
]


def clean_text(raw_text):
    text = raw_text.lower()
    text = text.replace(",", ".")  
    tokens = text.split()
    return tokens


# دالة التطابق التقريبي
def fuzzy_match(token, threshold=85):
    match, score, _ = process.extractOne(token, MEDICAL_KEYWORDS, scorer=fuzz.partial_ratio)
    if score >= threshold:
        return match
    return None




def extract_medical_values(tokens):
    results = {}
    for i, token in enumerate(tokens):
        matched_keyword = fuzzy_match(token)
        if matched_keyword:
          
            for j in range(i+1, min(i+4, len(tokens))):
                possible_value = tokens[j].replace('.', '', 1)
                if possible_value.replace(".", "", 1).isdigit():
                    results[matched_keyword.upper()] = tokens[j]
                    break
    return results