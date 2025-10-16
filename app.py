"""
DoctorFollow Medical Search Demo - Gradio Interface
Turkish Medical Literature RAG System with Dose Calculator
"""
import os
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

import gradio as gr

from rag import MedicalRAG
from dose_calculator import calculate_dose, get_supported_drugs

# Load environment variables
load_dotenv()

# Initialize RAG system
rag_system = MedicalRAG()

# Conversation history storage
conversation_history = []


def upload_pdf(file) -> str:
    """
    Handle PDF upload and indexing.

    Args:
        file: Uploaded file object from Gradio

    Returns:
        Status message
    """
    if file is None:
        return "❌ Lütfen bir PDF dosyası seçin."

    try:
        # Get file path
        pdf_path = file.name

        # Ingest PDF
        result = rag_system.ingest_pdf(pdf_path)

        if 'error' in result:
            return f"❌ Hata: {result['error']}"

        # Success message
        return f"""✅ PDF başarıyla yüklendi ve indekslendi!

📊 **İstatistikler:**
- Dosya: {result['document_name']}
- Toplam metin parçası: {result['total_chunks']}
- Toplam karakter: {result['total_characters']:,}
- Embedding boyutu: {result['embedding_dimensions']}

Artık sorularınızı sorabilirsiniz! 🎯"""

    except Exception as e:
        return f"❌ Hata: {str(e)}"


def chat_interface(message: str, history: List[Tuple[str, str]]) -> str:
    """
    Main chat interface with RAG system.

    Args:
        message: User message
        history: Gradio chat history format

    Returns:
        Assistant response
    """
    if not message.strip():
        return "Lütfen bir soru girin."

    # Convert Gradio history to our format
    conv_history = []
    for user_msg, assistant_msg in history:
        conv_history.append({
            'user': user_msg,
            'assistant': assistant_msg
        })

    # Query RAG system
    result = rag_system.ask(message, conversation_history=conv_history)

    # Build response
    response = result['answer']

    # Add sources if available
    if result.get('sources_formatted'):
        response += "\n\n" + result['sources_formatted']

    # Add validation info if citations are invalid
    if not result['citations_valid'] and result.get('citation_ids'):
        response += f"\n\n⚠️ {result['validation_message']}"

    # Update conversation history
    conversation_history.append({
        'user': message,
        'assistant': result['answer']
    })

    return response


def calculate_dose_interface(drug: str, weight: float, age: float) -> str:
    """
    Dose calculator interface.

    Args:
        drug: Drug name
        weight: Patient weight in kg
        age: Patient age in years

    Returns:
        Formatted dose calculation result
    """
    if not drug or not weight or not age:
        return "❌ Lütfen tüm alanları doldurun."

    try:
        result = calculate_dose(drug, weight, age)

        # Format output
        output = f"""## 💊 {result.drug_name}

### 📏 Hesaplanan Doz
**{result.dose}**

### ⏰ Kullanım Sıklığı
{result.frequency}

### 🧮 Hesaplama Yöntemi
{result.calculation_method}

### ⚠️ Uyarılar ve Notlar
"""
        for warning in result.warnings:
            output += f"- {warning}\n"

        # Safety indicator
        if result.is_safe:
            output += "\n✅ **Güvenlik Durumu:** Doz güvenli aralıkta görünüyor"
        else:
            output += "\n⚠️ **Güvenlik Durumu:** Lütfen hekim konsültasyonu yapın"

        return output

    except Exception as e:
        return f"❌ Hata: {str(e)}"


def get_system_stats() -> str:
    """Get and format system statistics."""
    stats = rag_system.get_stats()

    if not stats['indexed']:
        return "📊 **Sistem Durumu:** Henüz PDF yüklenmedi"

    return f"""📊 **Sistem İstatistikleri**

- Yüklü Doküman: {stats['document_name']}
- İndekslenmiş Parça: {stats['total_chunks']}
- Toplam Sorgu: {stats['total_queries']}
- Durum: ✅ Aktif
"""


def clear_conversation():
    """Clear conversation history."""
    global conversation_history
    conversation_history = []
    return []


# Build Gradio Interface
with gr.Blocks(
    title="DoctorFollow Medical Search Demo",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
    # 🏥 DoctorFollow Medical Search Demo

    **Türkçe Tıbbi Doküman Arama ve Doz Hesaplama Sistemi**

    Bu demo, tıbbi PDF'lerde akıllı arama ve pediatrik ilaç dozu hesaplama özellikleri sunar.
    """)

    with gr.Tabs():
        # Tab 1: PDF Upload & Search
        with gr.Tab("📄 Doküman Arama"):
            gr.Markdown("""
            ### PDF Yükleme ve Arama

            1. Bir tıbbi PDF yükleyin (örn: T.C. Sağlık Bakanlığı kılavuzları)
            2. PDF indekslendiğinde sorularınızı sorun
            3. Sistem kaynak atıflarıyla (citations) cevap verecektir
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    pdf_upload = gr.File(
                        label="PDF Dosyası Yükle",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    upload_btn = gr.Button("📤 Yükle ve İndeksle", variant="primary")
                    upload_status = gr.Textbox(
                        label="Yükleme Durumu",
                        lines=8,
                        interactive=False
                    )

                with gr.Column(scale=2):
                    stats_display = gr.Textbox(
                        label="Sistem İstatistikleri",
                        value=get_system_stats(),
                        lines=8,
                        interactive=False
                    )
                    refresh_stats_btn = gr.Button("🔄 İstatistikleri Yenile")

            gr.Markdown("### 💬 Soru-Cevap")

            chatbot = gr.Chatbot(
                label="DoctorFollow Asistan",
                height=400,
                show_label=True
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Sorunuzu yazın",
                    placeholder="Örnek: Çocuklarda parasetamol dozajı nedir?",
                    scale=4
                )
                send_btn = gr.Button("Gönder", variant="primary", scale=1)

            clear_btn = gr.Button("🗑️ Konuşmayı Temizle")

            gr.Markdown("""
            **Örnek Sorular:**
            - "Çocuklarda parasetamol dozajı nedir?"
            - "Amoksisilin ne zaman kullanılır?"
            - "Bu dozajı kaç saatte bir vermem gerekir?"
            """)

        # Tab 2: Dose Calculator
        with gr.Tab("💊 Doz Hesaplama"):
            gr.Markdown("""
            ### Pediatrik İlaç Doz Hesaplayıcı

            T.C. Sağlık Bakanlığı kılavuzlarına göre çocuk hastalarda doz hesaplama.

            **Desteklenen İlaçlar:** Amoksisilin, Parasetamol, İbuprofen
            """)

            with gr.Row():
                with gr.Column():
                    drug_input = gr.Dropdown(
                        choices=get_supported_drugs(),
                        label="İlaç Seçin",
                        value="Parasetamol"
                    )
                    weight_input = gr.Number(
                        label="Hasta Kilosu (kg)",
                        value=25,
                        minimum=0,
                        maximum=200
                    )
                    age_input = gr.Number(
                        label="Hasta Yaşı (yıl)",
                        value=7,
                        minimum=0,
                        maximum=18
                    )
                    calc_btn = gr.Button("🧮 Dozu Hesapla", variant="primary")

                with gr.Column():
                    dose_output = gr.Markdown(
                        label="Hesaplama Sonucu",
                        value="Lütfen ilaç seçin ve bilgileri girin."
                    )

            gr.Markdown("""
            ---
            **⚠️ UYARI:** Bu hesaplama yalnızca eğitim ve referans amaçlıdır.
            Gerçek hasta tedavisi için mutlaka hekim konsültasyonu yapılmalıdır.
            """)

        # Tab 3: About
        with gr.Tab("ℹ️ Hakkında"):
            gr.Markdown("""
            ## DoctorFollow Medical Search Demo

            ### 🎯 Özellikler

            - **Hibrit Arama:** BM25 (lexical) + Semantic (e5-small-v2) arama
            - **Kaynak Atıfları:** Vancouver tarzı tıbbi atıflar [1], [2], [3]
            - **Konuşma Hafızası:** Takip sorularını anlama
            - **Doz Hesaplama:** Pediatrik ilaç dozları güvenlik kontrolüyle
            - **Türkçe Optimizasyonu:** Türkçe tıbbi terimler için optimize edilmiş

            ### 🛠️ Teknoloji

            - **LLM:** AWS Bedrock Llama 3.1 8B Instruct
            - **Embeddings:** intfloat/e5-small-v2 (local)
            - **Search:** BM25 + Semantic + RRF Fusion
            - **Interface:** Gradio
            - **Cost:** $0 (AWS Free Tier + Local Models)

            ### 📚 Kaynak Format

            Bu sistem şu formatları destekler:
            - T.C. Sağlık Bakanlığı Kılavuzları (PDF)
            - Tıbbi protokoller ve yönergeler
            - Türkçe tıbbi literatür

            ### 🔒 Güvenlik

            - Tüm hesaplamalar referans amaçlıdır
            - Gerçek tedavi için hekim konsültasyonu gereklidir
            - Sistem sadece sağlanan dokümanlardaki bilgileri kullanır

            ### 👨‍💻 Geliştirici

            DoctorFollow Team - 2 Hour Demo Project

            ---

            **Versiyon:** 1.0.0 | **Tarih:** 2025
            """)

    # Event Handlers
    upload_btn.click(
        fn=upload_pdf,
        inputs=[pdf_upload],
        outputs=[upload_status]
    ).then(
        fn=get_system_stats,
        outputs=[stats_display]
    )

    refresh_stats_btn.click(
        fn=get_system_stats,
        outputs=[stats_display]
    )

    msg.submit(
        fn=chat_interface,
        inputs=[msg, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[msg]
    )

    send_btn.click(
        fn=chat_interface,
        inputs=[msg, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[msg]
    )

    clear_btn.click(
        fn=clear_conversation,
        outputs=[chatbot]
    )

    calc_btn.click(
        fn=calculate_dose_interface,
        inputs=[drug_input, weight_input, age_input],
        outputs=[dose_output]
    )


# Launch app
if __name__ == "__main__":
    print("🚀 Launching DoctorFollow Medical Search Demo...")
    print("📍 Local URL: http://localhost:7860")
    print("🌍 Public URL will be generated if share=True")
    print("\n⚠️  Make sure your .env file contains valid AWS credentials!\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public link
        show_error=True
    )
