import streamlit as st
import os
import json
import re
import sys
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- კონფიგურაცია ---
st.set_page_config(page_title="RS.GE InfoHub Agent", page_icon="🇬🇪")

raw_key = os.getenv("GROQ_API_KEY")
clean_key = "".join(char for char in raw_key if ord(char) < 128).strip()

PERSIST_DIRECTORY = "./infohub_vector_db"
JSON_PATH = "rs_ge_documents.json"
MANDATORY_CITATION = 'საინფორმაციო და მეთოდოლოგიური ჰაბზე განთავსებული დოკუმენტების მიხედვით - https://infohub.rs.ge/ka'

# --- მონაცემების ჩატვირთვა ---
@st.cache_resource
def get_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    if os.path.exists(PERSIST_DIRECTORY) and len(os.listdir(PERSIST_DIRECTORY)) > 0:
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

    if not os.path.exists(JSON_PATH):
        st.error("JSON ფაილი ვერ მოიძებნა!")
        return None

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    docs_objects = []
    for doc in raw_data.values():
        docs_objects.append(dict_to_doc(doc))

    return Chroma.from_documents(
        documents=docs_objects,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

@st.cache_data
def load_json():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# --- დამხმარე ფუნქცია ---
def dict_to_doc(info):
    """Генерирует объект Document из словаря JSON"""
    return Document(
        page_content=f"""დოკუმენტის ნომერი: {info.get('document_number', '')}
კატეგორია: {info.get('category', '')}
დოკუმენტის ტიპი: {info.get('document_type', '')}
მიღების თარიღი: {info.get('date', '')}""",
        metadata={
            "source": info.get("url", ""),
            "date": info.get("date", ""),
            "category": info.get("category", "")
        }
    )

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)

# --- STRICT ROUTER ფუნქციები ---
# --- ВСЕ ФУНКЦИИ ТЕПЕРЬ ВОЗВРАЩАЮТ СПИСОК (LIST) ---

def strict_search_by_url(query):
    data = load_json()
    for info in data.values():
        if info.get("url") and info["url"] in query:
            return [dict_to_doc(info)] # Даже один результат кладем в список
    return None

def strict_search_by_uuid(query):
    data = load_json()
    match = re.search(r"[0-9a-fA-F-]{36}", query)
    if not match: return None
    uuid = match.group(0)
    for info in data.values():
        if uuid in info.get("url", ""):
            return [dict_to_doc(info)]
    return None

def strict_search_by_number(query):
    data = load_json()
    # Улучшенный Regex: находит сложные номера (с буквами и символами)
    pattern = r"(?:[ა-ჰA-Za-z№-]*)\d+[\d/()-]*[ა-ჰA-Za-z]*"
    matches = re.findall(pattern, query)
    if not matches: return None

    results = []
    for doc_number in matches:
        for info in data.values():
            if info.get("document_number") == doc_number.strip():
                results.append(dict_to_doc(info))
    return results if results else None

def strict_search_by_exact_date(query):
    data = load_json()
    results = []
    month_pattern = r"(იანვ|თებერვ|მარტ|აპრილ|მაის|ივნის|ივლის|აგვისტ|სექტემბ|ოქტომბ|ნოემბ|დეკემბ)"
    
    # მომხმარებლის კითხვაში თვეების ფუძეზე დაყვანა (მარტის -> მარტ)
    query_base = re.sub(month_pattern + r"[ა-ჰ]*", r"\1", query.lower())

    for info in data.values():
        doc_date = info.get("date", "").lower()
        if not doc_date: continue
        
        # ბაზის თარიღის ფუძეზე დაყვანა (მარტი -> მარტ)
        doc_date_base = re.sub(month_pattern + r"[ა-ჰ]*", r"\1", doc_date)
        
        # თუ ფუძეებზე დაყვანილი თარიღი ემთხვევა
        if doc_date_base in query_base:
            results.append(dict_to_doc(info))
            
    return results if results else None

def strict_search_by_year(query):
    month_pattern = r"(იანვ|თებერვ|მარტ|აპრილ|მაის|ივნის|ივლის|აგვისტ|სექტემბ|ოქტომბ|ნოემბ|დეკემბ)"
    if re.search(month_pattern, query.lower()):
        return None 

    data = load_json()
    match = re.search(r"\b20\d{2}\b", query)
    if not match: return None
    
    year = match.group(0)
    results = [dict_to_doc(info) for info in data.values() if year in info.get("date", "")]
    return results if results else None


def strict_search_by_category(query):
    data = load_json()
    query_lc = query.lower()
    results = []
    for info in data.values():
        cat = info.get("category", "").lower()
        if cat and cat in query_lc:
            results.append(dict_to_doc(info))
    return results if results else None

def strict_search_by_year_range(query):
    month_pattern = r"(იანვ|თებერვ|მარტ|აპრილ|მაის|ივნის|ივლის|აგვისტ|სექტემბ|ოქტომბ|ნოემბ|დეკემბ)"
    
    # თუ თვეა ნახსენები, გამოვდივართ, რომ არ აირიოს ზუსტ თარიღებში
    if re.search(month_pattern, query.lower()):
        return None

    data = load_json()
    years = re.findall(r"\b20\d{2}\b", query)
    if len(years) < 2: return None

    start_year, end_year = int(min(years)), int(max(years))
    results = []

    for info in data.values():
        match = re.search(r"\b20\d{2}\b", info.get("date", ""))
        if match and start_year <= int(match.group(0)) <= end_year:
            results.append(dict_to_doc(info))
                
    return results if results else None

# --- HYBRID SEARCH ---
def hybrid_search(query, vector_db, k=4):
    if vector_db is None:
        return []       
    query_norm = normalize_text(query)
    data = load_json()
    scored_results = []

    for info in data.values():
        combined_text = f"{info.get('document_number', '')} {info.get('category', '')} {info.get('document_type', '')} {info.get('date', '')}"
        combined_text_norm = normalize_text(combined_text)
        score = 0

        if query_norm in combined_text_norm:
            score += 100

        for word in query_norm.split():
            if len(word) > 3 and word in combined_text_norm:
                score += 10

        if score > 0:
            scored_results.append((score, dict_to_doc(info)))

    scored_results.sort(key=lambda x: x[0], reverse=True)
    keyword_results = [doc for _, doc in scored_results[:k]]

    vector_results = vector_db.similarity_search(query, k=k)

    seen = set()
    final_results = []
    for doc in keyword_results + vector_results:
        key = doc.metadata.get("source")
        if key not in seen:
            seen.add(key)
            final_results.append(doc)

    return final_results[:k]

# --- ინიციალიზაცია ---
vector_db = get_vector_db()
llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0, 
    groq_api_key=clean_key
)

st.title("RS.GE InfoHub Agent 🤖")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("დასვით კითხვა (მაგ: აჩვენე დოკუმენტი 26385/2/2025)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("მიმდინარეობს ძებნა..."):
            
            # --- 1. STRICT ROUTER ლოგიკა ---
# --- 1. STRICT ROUTER ლოგიკა ---
# --- 1. STRICT ROUTER (ПОРЯДОК ПРИОРИТЕТА) ---
            strict_docs = (
                strict_search_by_url(prompt) or 
                strict_search_by_uuid(prompt) or   
                strict_search_by_number(prompt) or   
                strict_search_by_exact_date(prompt) or 
                strict_search_by_year_range(prompt) or  
                strict_search_by_year(prompt) or        
                strict_search_by_category(prompt)        
            )

            # თუ ნაპოვნია ზუსტი დამთხვევა (ერთი ან რამდენიმე)
            if strict_docs:
                response = "📌 **ნაპოვნია ზუსტი დამთხვევა ბაზაში:**\n\n"
                
                # ვაჩვენებთ პირველ 3 დოკუმენტს
                for d in strict_docs[:3]:
                    response += f"{d.page_content}\n\n🔗 **ბმული:** {d.metadata['source']}\n\n---\n"
                
                # თუ 3-ზე მეტია, ვამატებთ შეტყობინებას
                if len(strict_docs) > 3:
                    response += f"*(დამატებით მოიძებნა {len(strict_docs)-3} დოკუმენტი. გთხოვთ დააზუსტოთ მოთხოვნა)*\n\n"
                
                response += MANDATORY_CITATION
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.stop() # 🛑 პროცესი სრულდება, LLM არ ირთვება

            # --- 2. HYBRID + LLM ლოგიკა (თუ ზუსტი დამთხვევა ვერ მოიძებნა) ---
            found_docs = hybrid_search(prompt, vector_db, k=3)
            
            if not found_docs:
                response = f"მოთხოვნილი დოკუმენტი ვერ მოიძებნა საინფორმაციო და მეთოდოლოგიურ ჰაბზე.\n\n---\n{MANDATORY_CITATION}"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.stop()

            # Формируем контекст
            context_parts = []
            for i, d in enumerate(found_docs):
                context_parts.append(
                    f"--- დოკუმენტი {i+1} ---\nშინაარსი:\n{d.page_content}\n\nამ დოკუმენტის ბმული:\n{d.metadata['source']}\n"
                )
            context = "\n".join(context_parts)

            # Обновленный системный промпт (строгие правила)
            sys_msg = f"""შენ ხარ RS.GE-ს პროფესიონალი ასისტენტი.

მკაცრი წესები:
1. უპასუხე მხოლოდ ქართულად.
2. არ აურიო სხვადასხვა დოკუმენტის მონაცემები ერთმანეთში.
3. თუ ასახელებ დოკუმენტის ნომერს ან თარიღს — მიუთითე მხოლოდ ის ბმული, რომელიც ზუსტად ამ დოკუმენტს ეკუთვნის.
4. არ გამოიგონო ინფორმაცია (No hallucinations). თუ პასუხი კონტექსტში არ არის, თქვი რომ ინფორმაცია არ მოიპოვება.

კონტექსტი:
{context}"""
            
            try:
                res = llm.invoke([
                    {"role": "system", "content": sys_msg}, 
                    {"role": "user", "content": prompt}
                ])
                response = f"{res.content}\n\n---\n{MANDATORY_CITATION}"
            except Exception as e:
                response = f"⚠️ API შეცდომა (სცადეთ მოგვიანებით): {str(e)}"

            st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
