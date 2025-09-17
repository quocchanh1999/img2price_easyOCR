import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from rapidfuzz import process, fuzz
import unicodedata
import nltk
from nltk.stem.snowball import SnowballStemmer
import unicodedata
import easyocr
from PIL import Image
import io
import asyncio
import sys
import io
import google.generativeai as genai
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.set_page_config(layout="centered", page_title="Dự đoán Giá thuốc")

@st.cache_resource

def initialize_tools():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    stemmer = SnowballStemmer("english")
    reader = easyocr.Reader(['vi', 'en'], gpu=False) 

    return stemmer, reader

stemmer, ocr_reader = initialize_tools()

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(os.path.join(BASE_DIR, "final_model.joblib"))
        scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
        target_maps = joblib.load(os.path.join(BASE_DIR, "target_encoding_maps.joblib"))
        mean_price = joblib.load(os.path.join(BASE_DIR, "global_mean_price.joblib"))
        df_full = pd.read_excel(os.path.join(BASE_DIR, "dichvucong_medicines_Final.xlsx"))
        train_cols = pd.read_csv(os.path.join(BASE_DIR, "X_train_price_processed.csv")).columns.tolist()
        return model, scaler, target_maps, mean_price, df_full, train_cols
    except FileNotFoundError as e:
        st.error(f"Loaded file error.")
        st.stop()


REPLACEMENTS_DNSX = { 'ctcp': 'công ty cổ phần', 'tnhh': 'trách nhiệm hữu hạn', 'dp': 'dược phẩm', 'tw': 'trung ương', 'cty': 'công ty', 'ct': 'công ty', 'cp': 'cổ phần', 'sx': 'sản xuất', 'tm': 'thương mại', 'ld': 'liên doanh', 'mtv': 'một thành viên' }
GENERIC_TERMS_DNSX = [ 'công ty cổ phần', 'công ty tnhh', 'công ty', 'trách nhiệm hữu hạn', 'một thành viên', 'liên doanh', 'cổ phần', 'sản xuất', 'thương mại', 'trung ương', 'limited', 'ltd', 'pvt', 'inc', 'corp', 'corporation', 'gmbh', 'co', 'kg', 'ag', 'srl', 'international', 'pharma', 'pharmaceuticals', 'pharmaceutical', 'laboratories', 'industries' ]
GENERIC_TERMS_DNSX.sort(key=len, reverse=True)

def ultimate_company_name_cleaner(name_series, stemmer):
    def clean_single_name(name):
        name = str(name).lower()
        name = re.sub(r'\([^)]*\)', '', name)
        name = re.sub(r'[^a-z0-9\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        for old, new in REPLACEMENTS_DNSX.items():
            name = name.replace(old, new)
        tokens = name.split()
        stemmed_tokens = [stemmer.stem(token) for token in tokens if token]
        name = " ".join(stemmed_tokens)
        for term in GENERIC_TERMS_DNSX:
            stemmed_term = " ".join([stemmer.stem(t) for t in term.split()])
            if stemmed_term:
                name = name.replace(stemmed_term, '')
        return re.sub(r'\s+', ' ', name).strip()
    return name_series.apply(clean_single_name)

DEFINITIVE_DBC_MAP = {
    'Thuốc cấy/Que cấy': ['cấy dưới da', 'que cấy'],
    'Dạng xịt dưới lưỡi': ['xịt dưới lưỡi'],
    'Khí dung/Hít': ['khí dung', 'aerosol', 'inhaler', 'hít', 'phun mù'],
    'Thuốc đặt': ['thuốc đạn', 'viên đặt', 'đạn đặt', 'suppository', 'viên đạn'],
    'Thuốc gây mê đường hô hấp': ['gây mê', 'hô hấp'],
    'Trà túi lọc': ['trà túi lọc'],
    'Bột pha tiêm/truyền': ['bột đông khô pha tiêm', 'bột pha tiêm', 'powder for injection', 'bột và dung môi pha tiêm', 'bột đông khô', 'dung môi pha tiêm', 'bột vô khuẩn pha tiêm'],
    'Dung dịch tiêm/truyền': ['dung dịch tiêm', 'thuốc tiêm', 'bơm tiêm', 'injection', 'solution for injection', 'dịch truyền', 'dịch treo vô khuẩn', 'lọ', 'ống', 'dung dich tiêm'],
    'Hỗn dịch tiêm/truyền': ['hỗn dịch tiêm', 'suspension for injection'],
    'Nhũ tương tiêm/truyền': ['nhũ tương tiêm', 'emulsion for injection'],
    'Hoàn (YHCT)': ['hoàn mềm', 'hoàn cứng', 'viên hoàn'],
    'Cao lỏng (YHCT)': ['cao lỏng'],
    'Cao xoa/dán (YHCT)': ['cao xoa', 'cao dán'],
    'Dầu xoa/gió': ['dầu xoa', 'dầu gió', 'dầu xoa bóp', 'dầu bôi ngoài da'],
    'Kem bôi da': ['kem bôi', 'kem', 'cream'],
    'Gel bôi da': ['gel bôi', 'gel'],
    'Thuốc mỡ bôi da': ['thuốc mỡ', 'ointment', 'thuốc mỡ', 'mỡ bôi da', 'mỡ bôi ngoài da'],
    'Miếng dán': ['miếng dán', 'patch'],
    'Lotion': ['lotion'],
    'Cồn/Rượu thuốc': ['cồn thuốc', 'cồn xoa bóp', 'rượu thuốc'],
    'Nước súc miệng/Rơ miệng': ['nước súc miệng', 'rơ miệng'],
    'Dầu gội': ['dầu gội'],
    'Dung dịch nhỏ (Mắt/Mũi/Tai)': ['nhỏ mắt', 'nhỏ mũi', 'nhỏ tai', 'eye drops', 'nasal drops', 'dung dịch nhỏ mắt'],
    'Dung dịch xịt (Mũi/Tai)': ['xịt mũi', 'xịt', 'nasal spray', 'spray'],
    'Thuốc mỡ (Mắt/Mũi/Tai)': ['mỡ tra mắt', 'eye ointment'],
    'Viên nang': ['viên nang', 'nang', 'capsule', 'cap'],
    'Viên sủi': ['viên sủi', 'effervescent', 'cốm sủi bọt'],
    'Viên ngậm': ['viên ngậm', 'sublingual'],
    'Viên nén': ['viên nén', 'viên bao', 'tablet', 'nén bao', 'viên nhai', 'viên phân tán', 'viên'],
    'Siro': ['siro', 'sirô', 'siiro', 'syrup'],
    'Hỗn dịch uống': ['hỗn dịch uống', 'hỗn dịch', 'oral suspension', 'suspension'],
    'Nhũ tương uống': ['nhũ tương uống', 'nhũ tương', 'nhũ dịch uống', 'oral emulsion', 'nhỏ giọt'],
    'Dung dịch uống': ['dung dịch uống', 'oral solution', 'solution', 'thuốc nước uống', 'thuốc nước'],
    'Thuốc cốm uống': ['thuốc cốm', 'cốm pha', 'granules', 'cốm'],
    'Thuốc bột uống': ['thuốc bột pha uống', 'thuốc bột', 'powder', 'bột pha uống', 'bột'],
    'Dung dịch (Chung)': ['dung dịch'],
    'Dùng ngoài (Chung)': ['dùng ngoài', 'external', 'topical'],
    'Nguyên liệu': ['nguyên liệu', 'active ingredient'],
}

def classify_dangBaoChe_final(text):
    if pd.isnull(text): return "Không xác định"
    s = unicodedata.normalize('NFKC', str(text).lower())
    s = re.sub(r'[^a-z0-9à-ỹ\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    for standard_form, keywords in DEFINITIVE_DBC_MAP.items():
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', s) for keyword in keywords):
            return standard_form
    return 'Khác (Chưa phân loại)'

def extract_quantity(text):
    numbers = re.findall(r'(\d+[\.,]?\d*)', str(text))
    if not numbers:
        return 1.0
    numbers = [float(n.replace(',', '.')) for n in numbers]
    quantity = np.prod(numbers)
    return quantity if quantity > 0 else 1.0

ULTIMATE_UNIT_CONVERSION_MAP = { 'g': 1_000, 'mg': 1, 'mcg': 0.001, 'ml': 1_000 }
UNIT_REGEX = r'(\d+[\.,]?\d*)\s*(mcg|µg|mg|g|kg|ml|l|iu|ui|%)'

def extract_ingredient_features_ultimate(row):
    hoatChat_str = str(row.get('hoatChat', '')).lower().strip().replace('/', ';')
    hamLuong_val = row.get('hamLuong')
    hoatChat_list = [hc.strip() for hc in hoatChat_str.split(';')] if hoatChat_str else []
    so_luong_hoat_chat = len(hoatChat_list)
    hoat_chat_chinh = hoatChat_list[0] if so_luong_hoat_chat > 0 else "không rõ"
    if pd.isnull(hamLuong_val):
        return pd.Series([so_luong_hoat_chat, hoat_chat_chinh, 0.0, 0.0, 0.0], index=['so_luong_hoat_chat', 'hoat_chat_chinh', 'hl_chinh_mg', 'tong_hl_phu_mg', 'tong_hl_iu'])
    hamLuong_normalized = str(hamLuong_val).replace(',', '.')
    dosages = re.findall(UNIT_REGEX, hamLuong_normalized.lower())
    total_mg = 0.0
    total_iu = 0.0
    converted_dosages_mg = []
    for value_str, unit in dosages:
        value = float(value_str)
        if unit in ULTIMATE_UNIT_CONVERSION_MAP:
            converted_value = value * ULTIMATE_UNIT_CONVERSION_MAP[unit]
            converted_dosages_mg.append(converted_value)
        elif unit in ['iu', 'ui']:
            total_iu += value
    hl_chinh_mg = converted_dosages_mg[0] if len(converted_dosages_mg) > 0 else 0.0
    tong_hl_phu_mg = sum(converted_dosages_mg[1:]) if len(converted_dosages_mg) > 1 else 0.0
    return pd.Series([so_luong_hoat_chat, hoat_chat_chinh, hl_chinh_mg, tong_hl_phu_mg, total_iu], index=['so_luong_hoat_chat', 'hoat_chat_chinh', 'hl_chinh_mg', 'tong_hl_phu_mg', 'tong_hl_iu'])

def parse_user_query(query):
    parsed = {"tenThuoc": "N/A", "hoatChat": np.nan, "hamLuong": np.nan, "soLuong": "N/A", "donViTinh": "N/A"}
    temp_query = query
    match_hc = re.search(r'^(.*?)\s*\((.*?)\)', temp_query)
    if match_hc:
        parsed["hoatChat"] = match_hc.group(1).strip()  
        parsed["tenThuoc"] = match_hc.group(2).strip()
        temp_query = temp_query.replace(match_hc.group(0), parsed["tenThuoc"]) 
    else:
        parsed["tenThuoc"] = temp_query.strip() 

    hamluong_pattern = r'(\d+[\.,]?\d*\s*(?:mg|g|mcg|ml|l|iu|ui|kg)(?:\s*/\s*(?:ml|g|viên))?)'
    match_hl = re.search(hamluong_pattern, temp_query, re.IGNORECASE)
    if match_hl:
        parsed["hamLuong"] = match_hl.group(1).strip()
        temp_query = temp_query.replace(match_hl.group(0), '')

    unit_keywords = ['viên nang', 'viên nén', 'nang', 'viên', 'gói', 'ống', 'chai', 'lọ', 'hộp', 'tuýp']
    unit_pattern_sl = '|'.join(unit_keywords)
    match_sl = re.search(r'(\d+)\s*(' + unit_pattern_sl + r')\b', temp_query, re.IGNORECASE)
    if match_sl:
        parsed["soLuong"] = f"{match_sl.group(1)} {match_sl.group(2)}"
        parsed["donViTinh"] = match_sl.group(2).capitalize()
        temp_query = temp_query.replace(match_sl.group(0), '')

    if parsed["tenThuoc"] == "N/A":
        parsed["tenThuoc"] = temp_query.strip()
    
    parsed["quyCachDongGoi"] = parsed["soLuong"] if pd.notna(parsed["soLuong"]) else parsed["tenThuoc"]
    return parsed
    
def parse_dosage_value(hamLuong_val):
    if pd.isnull(hamLuong_val):
        return 0.0
    hamLuong_normalized = str(hamLuong_val).replace(',', '.')
    dosages = re.findall(UNIT_REGEX, hamLuong_normalized.lower())
    total_mg = 0.0
    for value_str, unit in dosages:
        value = float(value_str)
        if unit in ULTIMATE_UNIT_CONVERSION_MAP:
            total_mg += value * ULTIMATE_UNIT_CONVERSION_MAP[unit]
    return total_mg

def get_packaging_type(text):
    text = str(text).lower()
    if 'lọ' in text: return 'lọ'
    if 'chai' in text: return 'chai'
    if 'tuýp' in text or 'tube' in text: return 'tuýp'
    if 'ống' in text: return 'ống'
    if 'vỉ' in text: return 'vỉ'
    if 'hộp' in text: return 'hộp'
    if 'gói' in text: return 'gói'
    if 'túi' in text: return 'túi'
    return 'khác'

def get_base_unit(text):
    text = str(text).lower()
    if 'viên nang' in text or 'nang' in text: return 'nang'
    if 'viên' in text: return 'viên'
    if 'ml' in text: return 'ml'
    if 'g' in text or 'gam' in text: return 'g'
    if 'gói' in text: return 'gói'
    if 'ống' in text: return 'ống'
    return 'khác'


def transform_hybrid_data(hybrid_data, train_columns, target_maps, mean_price):
    df = pd.DataFrame([hybrid_data])

    df['doanhNghiepSanxuat_final'] = df['doanhNghiepSanxuat'].astype(str).str.lower().str.strip()
    df['nuocSanxuat_cleaned'] = df['nuocSanxuat'].astype(str).str.lower().str.strip()

    df['is_dangBaoChe_missing'] = df['dangBaoChe'].isnull().astype(int)
    df['dangBaoChe_final'] = df['dangBaoChe'].apply(classify_dangBaoChe_final)  
    df['soLuong'] = df['quyCachDongGoi'].apply(extract_quantity)
    df['loaiDongGoiChinh'] = df['quyCachDongGoi'].apply(get_packaging_type)
    df['donViCoSo'] = df['quyCachDongGoi'].apply(get_base_unit)

    ingredient_features = df.apply(extract_ingredient_features_ultimate, axis=1)
    df = pd.concat([df, ingredient_features], axis=1)

    column_name_mapping = {
        'doanhNghiepSanxuat': 'doanhNghiepSanxuat_final',
        'nuocSanxuat': 'nuocSanxuat_cleaned',
        'dangBaoChe': 'dangBaoChe_final',
        'hoat_chat_chinh': 'hoat_chat_chinh',
        'loaiDongGoiChinh': 'loaiDongGoiChinh',
        'donViCoSo': 'donViCoSo'
    }

    for original_col, final_col in column_name_mapping.items():
        if original_col in target_maps:
            target_map = target_maps[original_col]

            series_to_encode = df[final_col].astype(str).str.lower().str.strip()

            encoded_col_name = final_col + '_encoded'
            df[encoded_col_name] = series_to_encode.map(target_map).fillna(mean_price)
        else:
            pass
    df_final = df.reindex(columns=train_columns)
    df_final.fillna(0, inplace=True)
    
    return df_final

@st.cache_data

def call_gemini_parser(ocr_text):

    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash-lite')

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        prompt = f"""
        Giờ tôi gửi bạn một đoạn OCR từ toa thuốc. Nhiệm vụ của bạn là liệt kê những thuốc trong đó ra theo danh sách tương ứng như sau:

        - Tên thuốc:
        - Hoạt chất:
        - Hàm lượng:
        - Số lượng:
        - Đơn vị tính:

        Một ví dụ như sau:

        1. Tên thuốc: UNASYN
        Hoạt chất: Sultamicillin
        Hàm lượng: 375mg
        Số lượng: 8 viên
        Đơn vị tính: Viên

        2. Tên thuốc: NEXT G CAL
        Hoạt chất: (Chưa xác định)
        Hàm lượng: (Chưa xác định)
        Số lượng: 30 viên
        Đơn vị tính: Viên

        3. Tên thuốc: HEMO Q MOM,
        Hoạt chất: (Chưa xác định),
        Hàm lượng: (Chưa xác định),
        Số lượng: 30 Viên,
        Đơn vị: Viên

        4. Tên thuốc: POVIDINE,
        Hoạt chất: (Chưa xác định),
        Hàm lượng: 10% 90ML,
        Số lượng: 1 Chai,
        Đơn vị: Chai

        5. Tên thuốc: Bộ chăm sóc rốn,
        Hoạt chất: (Chưa xác định),
        Hàm lượng: (Chưa xác định),
        Số lượng: 1 Bộ,
        Đơn vị: Bộ


        Ở những thuốc có dấu (), ví dụ như Gliclazid (Staclazide 30 MR) thì phần trong dấu () là tên thuốc, phần ở ngoài là hoạt chất, như trường hợp này thì tên thuốc là Staclazide 30 MR, hoạt chất là Gliclazid. không được phép lấy cả Gliclazid (Staclazide 30 MR) cho tên thuốc. 

        Còn những trường hợp chỉ có chữ in hoa như NEXT G CAL thì NEXT G CAL là tên thuốc luôn. chữ O ngay trước đơn vị như mg là số 0 bị OCR nhầm.

        Tuy nhiên, nếu thuốc có () và có chứa cả chữ in hoa toàn bộ, như YESOM 40 40mg (Esomeprazol 40mg) thì phần trong ngoặc mới là hoạt chất, còn phần ia hoa toàn bộ luôn luôn là tên thuốc. Như trong trường hợp này thì YESOM 40 là tên thuôc, Esomeprazol là hoạt chất.

        Đôi khi kết quả OCR cũng sẽ bị sai chính tả, ví dụ 'Cac lonỉ vitamin' thì bạn sửa lại cho đúng là 'Các loại vitamin'.
        
        Bạn hiểu chưa, và chỉ trả lời bằng cách liệt kê ra tên thuốc và các giá trị tương ứng theo đúng định dạng mà ví dụ tôi đã cung cấp. không trả lời bất cứ gì khác. chỉ trả lời phần cần thiết không giải thích bất cứ gì khác, không mở ngoặc đề chú thích. phải trả lời theo dạng liệt kê xuống dòng như trên. Nhớ là phải liệt kê đủ số lượng đấy nhé.
        OCR đây:
        ---
        {ocr_text}
        ---
        """
        
        response = model.generate_content(prompt, safety_settings=safety_settings)
        return response.text
    except Exception as e:
        st.error(f"Lỗi khi gọi API của Google AI: {e}")
        return None

def parse_gemini_response(response_text):

    parsed_drugs = []

    blocks = re.split(r'\n*(?=[-\d\.]*\s*Tên thuốc:)', response_text)
    
    for block in blocks:
        if not block.strip():
            continue

        ten_thuoc = re.search(r"Tên thuốc:\s*(.*)", block)
        hoat_chat = re.search(r"Hoạt chất:\s*(.*)", block)
        ham_luong = re.search(r"Hàm lượng:\s*(.*)", block)
        so_luong = re.search(r"Số lượng:\s*(.*)", block)
        don_vi_tinh = re.search(r"Đơn vị tính:\s*(.*)", block)

        drug_dict = {
            "tenThuoc": ten_thuoc.group(1).strip() if ten_thuoc and "(Chưa xác định)" not in ten_thuoc.group(1) else np.nan,
            "hoatChat": hoat_chat.group(1).strip() if hoat_chat and "(Chưa xác định)" not in hoat_chat.group(1) else np.nan,
            "hamLuong": ham_luong.group(1).strip() if ham_luong and "(Chưa xác định)" not in ham_luong.group(1) else np.nan,
            "soLuong": so_luong.group(1).strip() if so_luong and "(Chưa xác định)" not in so_luong.group(1) else np.nan,
            "donViTinh": don_vi_tinh.group(1).strip() if don_vi_tinh and "(Chưa xác định)" not in don_vi_tinh.group(1) else np.nan,
        }

        if pd.notna(drug_dict["tenThuoc"]) or pd.notna(drug_dict["hoatChat"]):
            parsed_drugs.append(drug_dict)
        
    return parsed_drugs


st.title("Gợi ý Giá thuốc")

model, scaler, target_maps, mean_price, df_full, train_cols = load_artifacts()

if df_full is not None:
    user_query_text = st.text_input("", placeholder="Nhập tên thuốc, ví dụ Hoạt chất (tên thuốc) hàm lượng số lượng...", label_visibility="collapsed")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    query_to_process = None
    source = "text"

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        with st.spinner("Đang đọc ảnh..."):
            ocr_text = " ".join(ocr_reader.readtext(image_bytes, detail=0))
        query_to_process = ocr_text
        source = "ocr"
        with st.expander("Xem toàn bộ văn bản nhận dạng được"):
            st.text_area("", ocr_text, height=150)
    elif user_query_text:
        query_to_process = user_query_text
        
    if query_to_process:
        st.markdown("---")
        with st.spinner("AI đang phân tích đơn thuốc..."):
            if source == "ocr":
                structured_response = call_gemini_parser(query_to_process)
                if not structured_response:
                    st.error("AI không thể phân tích văn bản từ ảnh này.")
                    st.stop()
                lines_to_process = parse_gemini_response(structured_response)
            else:
                lines_to_process = [parse_user_query(query_to_process)]
        
        if not lines_to_process:
            st.warning("Không nhận dạng được đơn thuốc nào.")
            st.stop()

        valid_drug_count = 0
        for i, parsed_info in enumerate(lines_to_process):
            search_key = f"{parsed_info.get('tenThuoc','')} {parsed_info.get('hoatChat','')}".strip()
            if not search_key or (pd.isna(parsed_info.get('tenThuoc')) and pd.isna(parsed_info.get('hoatChat'))):
                continue  
            
            valid_drug_count += 1
            if len(lines_to_process) > 1:
                st.markdown(f"--- \n ### 💊 Kết quả cho đơn thuốc {valid_drug_count}")

            with st.spinner(f"Đang xử lý: '{search_key[:50]}...'"):
                st.markdown(f"**Tên thuốc:** {parsed_info.get('tenThuoc') or '(Chưa xác định)'}")
                st.markdown(f"**Hoạt chất:** {parsed_info.get('hoatChat') or '(Chưa xác định)'}")
                st.markdown(f"**Hàm lượng:** {parsed_info.get('hamLuong') or '(Chưa xác định)'}")
                st.markdown(f"**Số lượng:** {parsed_info.get('soLuong') or 'N/A'}")
                st.markdown(f"**Đơn vị tính:** {parsed_info.get('donViTinh') or 'N/A'}")
                st.markdown("---")

                choices = df_full['tenThuoc'].dropna().tolist()
                best_match, score, _ = process.extractOne(search_key, choices)
                
                if not best_match:
                    st.warning("Không tìm thấy thuốc tương tự trong CSDL.")
                else:
                    drug_info_row = df_full[df_full['tenThuoc'] == best_match].iloc[0]

                    if score >= 90:
                        can_extrapolate = False
                        if pd.notna(parsed_info.get('hoatChat')) and pd.notna(parsed_info.get('hamLuong')):
                            user_hc_clean = str(parsed_info.get('hoatChat', '')).lower()
                            db_hc_clean = str(drug_info_row.get('hoatChat', '')).lower()
                            if fuzz.partial_ratio(user_hc_clean, db_hc_clean) > 80:
                                user_dosage_mg = parse_dosage_value(parsed_info.get('hamLuong'))
                                db_dosage_mg = parse_dosage_value(drug_info_row.get('hamLuong'))
                                if user_dosage_mg > 0 and db_dosage_mg > 0 and user_dosage_mg != db_dosage_mg:
                                    can_extrapolate = True
                        
                        if can_extrapolate:
                            ratio = user_dosage_mg / db_dosage_mg
                            st.markdown(f"**Phương thức:** `Ngoại suy theo hàm lượng (Tỷ lệ: {ratio:.2f}x)`")
                            st.caption(f"Dựa trên giá của *{best_match}* (độ tương đồng tên: {score:.0f}%)")
                            gia_kk_base = drug_info_row['giaBanBuonDuKien']
                            gia_tt_base = drug_info_row.get('giaThanh', np.nan)
                            gia_kk_extrapolated = gia_kk_base * ratio
                            gia_tt_extrapolated = gia_tt_base * ratio if pd.notna(gia_tt_base) else np.nan
                            st.metric("Giá Kê Khai (Ước tính)", f"{gia_kk_extrapolated:,.0f} VND" if pd.notna(gia_kk_base) else "Không có dữ liệu")
                            st.metric("Giá Thị Trường (Ước tính)", f"{gia_tt_extrapolated:,.0f} VND" if pd.notna(gia_tt_base) else "Không có dữ liệu")
                        else:
                            st.markdown(f"**Phương thức:** `Levenshtein distance (Thuốc tương đồng: {best_match}, độ tương đồng: {score:.0f}%)`")
                            gia_kk = drug_info_row['giaBanBuonDuKien']
                            gia_tt = drug_info_row.get('giaThanh', np.nan)
                            st.metric("Giá Kê Khai", f"{gia_kk:,.0f} VND" if pd.notna(gia_kk) else "Không có dữ liệu")
                            st.metric("Giá Thị Trường", f"{gia_tt:,.0f} VND" if pd.notna(gia_tt) else "Không có dữ liệu")

                    else:
                        st.markdown(f"**Phương thức:** `XGBoost Regressor`")
                        st.caption(f"Sử dụng thông tin bổ sung (nhà SX, nước SX, Dạng bào chế...) từ thuốc tương tự nhất: *{best_match}*")
                        
                        hybrid_data = {
                            'hoatChat': parsed_info.get('hoatChat') or drug_info_row.get('hoatChat'),
                            'hamLuong': parsed_info.get('hamLuong') or drug_info_row.get('hamLuong'),
                            'quyCachDongGoi': parsed_info.get('quyCachDongGoi') or drug_info_row.get('quyCachDongGoi'),
                            'doanhNghiepSanxuat': drug_info_row.get('doanhNghiepSanxuat'),
                            'nuocSanxuat': drug_info_row.get('nuocSanxuat'),
                            'dangBaoChe': drug_info_row.get('dangBaoChe')
                        }
                        
                        try:
                            transformed_data = transform_hybrid_data(hybrid_data, train_cols, target_maps, mean_price)
                            scaled_data = scaler.transform(transformed_data)
                            prediction_log = model.predict(scaled_data)
                            prediction = np.expm1(prediction_log)
                            gia_kk_pred, gia_tt_pred = prediction[0][0], prediction[0][1]
                            st.metric("Giá Kê Khai (Dự đoán)", f"{gia_kk_pred:,.0f} VND")
                            st.metric("Giá Thị Trường (Dự đoán)", f"{gia_tt_pred:,.0f} VND")
                        except Exception as e:
                            st.error(f"Lỗi khi dự đoán: {e}")
