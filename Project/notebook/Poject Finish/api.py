from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForSequenceClassification, pipeline
from serpapi import GoogleSearch
from deep_translator import GoogleTranslator
import uvicorn

# ========== Load Model ==========
tokenizer_t5 = T5Tokenizer.from_pretrained("oke190231/t5-tokenizer-50percent")
model_t5 = T5ForConditionalGeneration.from_pretrained("oke190231/t5-squad2-checkpoint")

tokenizer_bert = BertTokenizer.from_pretrained("oke190231/bert_intent_model")
model_bert = BertForSequenceClassification.from_pretrained("oke190231/bert_intent_model")
classifier = pipeline("text-classification", model=model_bert, tokenizer=tokenizer_bert)

# ========== Intent Mapping ==========
intent_labels = {
    "kata-kasar": 0,
    "laporan-kekerasan": 1,
    "psikologi": 2,
    "data-umum": 3,
    "jumlah-kdrt": 4
}
reverse_labels = {v: k for k, v in intent_labels.items()}

# ========== Fallback Context (Opsional) ==========
fallback_contexts = {
    "depresi": "Jika kamu sedang merasa depresi, cobalah untuk bicara dengan orang terpercaya, beristirahat cukup, dan pertimbangkan untuk konsultasi ke psikolog.",
    "kdrt": "Kekerasan dalam rumah tangga adalah tindakan pidana. Segera hubungi lembaga perlindungan seperti Komnas Perempuan atau pihak berwajib."
}

# ========== FastAPI App ==========
app = FastAPI()

# ========== Request Schema ==========
class QuestionInput(BaseModel):
    question: str

# ========== Fungsi Intent ==========
def predict_intent(texts):
    predictions = classifier(texts)
    results = []
    for text, pred in zip(texts, predictions):
        label_index = int(pred["label"].split("_")[1])
        intent = reverse_labels[label_index]
        results.append((text, intent))
    return results

# ========== Fungsi Cari Konteks ==========
def search_context(query):
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": "d8184b379a4a7abdaee377686c40c26ea1a9841fa9436735f9a11064c0ccf3eb",
            "num": 5,
            "hl": "id"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        context_list = []

        if "organic_results" in results:
            for r in results["organic_results"]:
                if "snippet" in r:
                    context_list.append(r["snippet"])

        if not context_list:
            for keyword in fallback_contexts:
                if keyword in query.lower():
                    return fallback_contexts[keyword]
            return "Context not found."

        combined_context = " ".join(context_list)
        # Terjemahkan hasil konteks ke Inggris
        translator = GoogleTranslator(source="auto", target="en")
        translated_context = translator.translate(combined_context)
        return translated_context

    except Exception as e:
        return f"Error SerpAPI: {str(e)}"

# ========== Fungsi Jawab Pertanyaan ==========
def generate_answer(question, context):
    """Jawab pertanyaan dengan model T5. Terjemahkan pertanyaan ke Inggris, hasilnya ke Indonesia."""
    if "Error" in context or "Context not found" in context:
        return "Maaf, aku tidak bisa menemukan informasi yang relevan."

    # Terjemahkan pertanyaan ke Inggris
    translated_question = GoogleTranslator(source="auto", target="en").translate(question)

    # Format input untuk T5
    input_text = f"question: {translated_question}  context: {context}"
    input_ids = tokenizer_t5(input_text, return_tensors="pt").input_ids

    # Generate jawaban
    output_ids = model_t5.generate(input_ids, max_length=128)
    english_answer = tokenizer_t5.decode(output_ids[0], skip_special_tokens=True)

    # Terjemahkan jawaban ke Bahasa Indonesia
    indonesian_answer = GoogleTranslator(source="en", target="id").translate(english_answer)
    return indonesian_answer


# ========== Endpoint API ==========
@app.post("/ask")
def ask_question(data: QuestionInput):
    question = data.question
    intent = predict_intent([question])[0][1]
    context = search_context(question)
    answer = generate_answer(question, context)

    return {
        "question": question,
        "intent": intent,
        "context": context,
        "answer": answer
    }

# ========== Jalankan Server ==========
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


# ============buat run filenya ============
# uvicorn app:app --reload  
