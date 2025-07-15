from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import uuid
import os
from ocr import extract_text
from parser import clean_text
from parser import extract_medical_values
from utils import normalize_keys_fuzzy
from predicer import predict_dynamic
from report import generate_report


mapping = {
    "glucose": ["glu", "Glu", "Glucose", "gloucose", "glucose", "GLUCOSE", "Glu (fasting)", "blood sugar"],
    "urea": ["urea", "UREA", "Urea", "BUN", "bun"],
    "estrogen": ["estrogen", "ESTROGEN", "Estrogen", "Estradiol", "E2", "estradiol"],
    "creatinine": ["creatinine", "CREATININE", "Creatinine", "creat", "Creat"],
    "calcium": ["calcium", "CALCIUM", "Calcium", "ca", "Ca", "Ca2+"],
    "total_cholesterol": ["total cholesterol", "Total Cholesterol", "cholesterol", "Cholesterol", "TOTAL CHOLESTEROL", "TC", "Chol"],
    "cholesterol": ["cholesterol", "Cholesterol", "TC", "total cholesterol"],  # مرادف
    "lipoprotein": ["lipoprotein", "Lipoprotein", "LIPOPROTEIN", "Lp(a)", "LPA"],
    "albumin": ["albumin", "Albumin", "ALBUMIN", "Alb"],
    "bilirubin": ["bilirubin", "Bilirubin", "BILIRUBIN", "T.Bil", "Total Bilirubin", "TBIL"],
    "potassium": ["k", "K", "potassium", "Potassium", "K+"],
    "ph": ["ph", "PH", "Ph", "Blood pH", "pH (blood)"],
    "alt": ["alt", "ALT", "sgpt", "SGPT", "Alt", "ALT (SGPT)"],
    "ast": ["ast", "AST", "sgot", "SGOT", "Ast", "AST (SGOT)"],
    "magnesium": ["mg", "MG", "magnesium", "Magnesium", "Mg2+"],
    "ldl": ["ldl", "Ldl", "LDL", "LDL-C"],
    "hdl": ["hdl", "HDL", "HDL-C"],
    "triglycerides": ["triglycerides", "Triglycerides", "tri", "Trigly", "TG", "TG (Triglycerides)"],
    "hemoglobin": ["hemoglobin", "HEMOGLOBIN", "Hb", "hb", "HGB"],
    "hdl": ["hdl", "HDL", "HDL-C"],
    "bun": ["BUN", "bun", "Urea Nitrogen", "Blood Urea Nitrogen"],
    "wbc": ["WBC", "wbc", "White Blood Cells", "Leukocytes", "Leucocytes"],
    "rbc": ["RBC", "rbc", "Red Blood Cells", "Erythrocytes"],
    "platelets": ["platelets", "PLT", "Platelet Count", "Thrombocytes"],
    "uric_acid": ["UA", "ua", "Uric Acid", "uric acid", "URIC ACID"]

}






interpretations = {
    "urea": {
        "name": "اليوريا (حمض البول)",
        "أقل من الحد الطبيعي": "منخفضة، وهذا قد يشير إلى سوء تغذية أو أمراض كبد.",
        "قيمة طبيعية نوعا ما ": "ضمن المعدل الطبيعي، مما يدل على كفاءة الكلى.",
        "أعلى من الحد الطبيعي": "مرتفعة، وقد تشير إلى خلل في وظائف الكلى أو جفاف."
    },
    "magnesium": {
        "name": "المغنيسيوم",
        "أقل من الحد الطبيعي": "منخفض، وقد يسبب تشنج عضلي أو إرهاق.",
        "قيمة طبيعية نوعا ما ": "في الحدود الطبيعية، مما يدل على توازن جيد في المعادن.",
        "أعلى من الحد الطبيعي": "مرتفع، وقد يدل على مشاكل في الكلى أو تناول مكملات زائدة."
    },
    "glucose": {
        "name": "الجلوكوز (السكر)",
        "أقل من الحد الطبيعي": "منخفض، وقد يدل على نقص السكر في الدم.",
        "قيمة طبيعية نوعا ما ": "طبيعي، وهذا مؤشر جيد للتحكم في السكر.",
        "أعلى من الحد الطبيعي": "مرتفع، وقد يشير إلى مقاومة إنسولين أو سكري."
    },
    "ldl": {
        "name": "الكوليسترول الضار (LDL)",
        "أقل من الحد الطبيعي": "منخفض، وهذا جيد لصحة القلب.",
        "قيمة طبيعية نوعا ما ": "في المعدلات المقبولة، حافظ عليه.",
        "أعلى من الحد الطبيعي": "مرتفع، وهذا قد يزيد خطر أمراض القلب."
    },
    "calcium": {
        "name": "الكالسيوم",
        "أقل من الحد الطبيعي": "منخفض، وقد يسبب هشاشة عظام أو تنميل.",
        "قيمة طبيعية نوعا ما ": "ضمن المعدل الطبيعي، وهذا جيد.",
        "أعلى من الحد الطبيعي": "مرتفع، وقد يشير إلى مشاكل في الغدد الجار درقية."
    },
    "estrogen": {
        "name": "الإستروجين",
        "أقل من الحد الطبيعي": "منخفض، وقد يدل على خلل هرموني أو مشاكل في الخصوبة.",
        "قيمة طبيعية نوعا ما ": "ضمن المعدلات الطبيعية.",
        "أعلى من الحد الطبيعي": "مرتفع، وقد يرتبط بأكياس المبيض أو الحمل."
    },
    "creatinine": {
        "name": "الكرياتينين",
        "أقل من الحد الطبيعي": "منخفض، قد يشير إلى ضعف في العضلات.",
        "قيمة طبيعية نوعا ما ": "طبيعي، مما يدل على وظائف كلى جيدة.",
        "أعلى من الحد الطبيعي": "مرتفع، مما قد يدل على خلل في الكلى."
    },
    "total_cholesterol": {
        "name": "الكوليسترول الكلي",
        "أقل من الحد الطبيعي": "منخفض، قد يشير إلى سوء تغذية.",
        "قيمة طبيعية نوعا ما ": "ضمن المعدل الطبيعي.",
        "أعلى من الحد الطبيعي": "مرتفع، وقد يزيد من خطر أمراض القلب."
    },
    "lipoprotein": {
        "name": "الليبوبروتين (a)",
        "أقل من الحد الطبيعي": "منخفض، وهو أمر جيد لصحة القلب.",
        "قيمة طبيعية نوعا ما ": "طبيعي.",
        "أعلى من الحد الطبيعي": "مرتفع، وقد يزيد من خطر أمراض القلب."
    },
    "albumin": {
        "name": "الألبومين",
        "أقل من الحد الطبيعي": "منخفض، وقد يدل على مشاكل كبدية أو سوء تغذية.",
        "قيمة طبيعية نوعا ما ": "ضمن الطبيعي.",
        "أعلى من الحد الطبيعي": "مرتفع، قد يدل على جفاف."
    },
    "bilirubin": {
        "name": "البيليروبين",
        "أقل من الحد الطبيعي": "منخفض، نادراً ما يكون مقلقًا.",
        "قيمة طبيعية نوعا ما ": "طبيعي.",
        "أعلى من الحد الطبيعي": "مرتفع، وقد يدل على مشاكل كبد أو انسداد في القنوات الصفراوية."
    },
    "potassium": {
        "name": "البوتاسيوم",
        "أقل من الحد الطبيعي": "منخفض، قد يسبب ضعف أو اضطراب نبض القلب.",
        "قيمة طبيعية نوعا ما ": "ضمن الطبيعي.",
        "أعلى من الحد الطبيعي": "مرتفع، وقد يسبب اضطراب في القلب."
    },
    "ph": {
        "name": "درجة الحموضة (pH)",
        "أقل من الحد الطبيعي": "حمضي، قد يدل على حماض استقلابي.",
        "قيمة طبيعية نوعا ما ": "في النطاق الطبيعي.",
        "أعلى من الحد الطبيعي": "قلوي، قد يدل على قلونة الدم."
    },
    "alt": {
        "name": "إنزيم ALT",
        "أقل من الحد الطبيعي": "منخفض، نادرًا ما يكون مقلقًا.",
        "قيمة طبيعية نوعا ما ": "طبيعي.",
        "أعلى من الحد الطبيعي": "مرتفع، وقد يشير إلى ضرر في الكبد."
    },
    "ast": {
        "name": "إنزيم AST",
        "أقل من الحد الطبيعي": "منخفض، غير مقلق عادة.",
        "قيمة طبيعية نوعا ما ": "طبيعي.",
        "أعلى من الحد الطبيعي": "مرتفع، وقد يدل على مشاكل في الكبد أو العضلات."
    },
    "triglycerides": {
        "name": "الدهون الثلاثية",
        "أقل من الحد الطبيعي": "منخفضة، أمر غير شائع وقد يدل على سوء تغذية.",
        "قيمة طبيعية نوعا ما ": "طبيعي.",
        "أعلى من الحد الطبيعي": "مرتفعة، وقد ترتبط بأمراض القلب أو السكري."
    },
    "hemoglobin": {
        "name": "الهيموغلوبين",
        "أقل من الحد الطبيعي": "منخفض، وقد يدل على فقر دم.",
        "قيمة طبيعية نوعا ما ": "طبيعي.",
        "أعلى من الحد الطبيعي": "مرتفع، وقد يشير إلى نقص الأكسجين أو مشاكل رئوية."
    },
    "hdl": {
        "name": "الكوليسترول الجيد (HDL)",
        "أقل من الحد الطبيعي": "منخفض، مما قد يزيد خطر أمراض القلب.",
        "قيمة طبيعية نوعا ما ": "طبيعي.",
        "أعلى من الحد الطبيعي": "مرتفع، وهو أمر مفيد للقلب."
    },
    "cholesterol": {
        "name": "الكوليسترول الكلي",
        "أقل من الحد الطبيعي": "منخفض، قد يشير إلى سوء تغذية.",
        "قيمة طبيعية نوعا ما ": "طبيعي.",
        "أعلى من الحد الطبيعي": "مرتفع، وقد يزيد من خطر أمراض القلب."
    },
    "bun": {
        "name": "النيتروجين يوريا الدم (BUN)",
        "أقل من الحد الطبيعي": "منخفض، قد يدل على أمراض كبد أو سوء تغذية.",
        "قيمة طبيعية نوعا ما ": "طبيعي.",
        "أعلى من الحد الطبيعي": "مرتفع، قد يشير إلى مشاكل في الكلى."
    },
    "wbc": {
        "name": "كريات الدم البيضاء",
        "أقل من الحد الطبيعي": "منخفضة، قد تدل على ضعف مناعي.",
        "قيمة طبيعية نوعا ما ": "ضمن الطبيعي.",
        "أعلى من الحد الطبيعي": "مرتفعة، قد تشير إلى التهاب أو عدوى."
    },
    "rbc": {
        "name": "كريات الدم الحمراء",
        "أقل من الحد الطبيعي": "منخفضة، قد تدل على فقر دم.",
        "قيمة طبيعية نوعا ما ": "طبيعي.",
        "أعلى من الحد الطبيعي": "مرتفعة، قد تدل على نقص أكسجة مزمن."
    },
    "platelets": {
        "name": "الصفائح الدموية",
        "أقل من الحد الطبيعي": "منخفضة، قد تؤدي إلى نزيف.",
        "قيمة طبيعية نوعا ما ": "طبيعي.",
        "أعلى من الحد الطبيعي": "مرتفعة، قد تشير إلى التهابات أو اضطرابات دموية."
    },
        "uric_acid": {
        "name": "حمض اليوريك (UA)",
        "أقل من الحد الطبيعي": "منخفض، قد يشير إلى مشاكل بالكبد أو سوء تغذية.",
        "قيمة طبيعية نوعا ما ": "في الحدود الطبيعية، وهذا جيد.",
        "أعلى من الحد الطبيعي": "مرتفع، قد يشير إلى النقرس أو مشاكل كلوية."
    }

}



app = FastAPI()

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    file_ext = file.filename.split(".")[-1]
    temp_filename = f"temp_{uuid.uuid4()}.{file_ext}"

    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
       
        text_data = extract_text(temp_filename)
        if isinstance(text_data, list):
           text_data = " ".join(text_data)
        tokens = clean_text(text_data)  # clean text 
        results = extract_medical_values(tokens) 
      
        normalized_data = normalize_keys_fuzzy(results, mapping)
        
        predictions = predict_dynamic(normalized_data)
        
        report = generate_report(predictions , interpretations=interpretations)
          
        return JSONResponse(content={"analyzed_results":report})
    finally:
        os.remove(temp_filename)
