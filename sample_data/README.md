# Sample Data Folder

## Purpose
This folder is for storing Turkish medical PDF documents that will be indexed by the DoctorFollow system.

## Recommended Sample Documents

For testing this demo, download Turkish medical guidelines from official sources:

### T.C. Sağlık Bakanlığı (Ministry of Health) Resources

1. **Antibiyotik Kullanım Kılavuzu**
   - Source: https://hsgm.saglik.gov.tr/
   - Topics: Antibiotic usage guidelines, dosing, resistance

2. **Çocuk Hastalıkları Protokolleri**
   - Source: https://hsgm.saglik.gov.tr/
   - Topics: Pediatric disease management protocols

3. **Aşı Kılavuzu**
   - Source: https://asi.saglik.gov.tr/
   - Topics: Vaccination schedules and guidelines

## File Requirements

- **Format:** PDF only
- **Language:** Turkish (optimized) or English
- **Content:** Medical guidelines, protocols, drug information
- **Size:** Recommended < 10MB for fast processing

## How to Use

1. Download a Turkish medical PDF from the sources above
2. Place the PDF file in this `sample_data/` folder
3. Launch the DoctorFollow app: `python app.py`
4. Use the PDF upload interface to index the document
5. Start asking questions!

## Example Questions to Test

Once you've uploaded a medical guideline PDF, try these questions:

### General Medical Queries (Turkish)
- "Çocuklarda parasetamol dozajı nedir?"
- "Amoksisilin ne zaman kullanılır?"
- "Ateş düşürücü ilaçların yan etkileri nelerdir?"

### Follow-up Questions
- "Bu dozajı kaç saatte bir vermem gerekir?"
- "Maksimum günlük doz nedir?"
- "Hangi yaş grubunda kontrendike?"

### Dose Calculator Tests
- Drug: Amoksisilin, Weight: 25kg, Age: 7 years
- Drug: Parasetamol, Weight: 15kg, Age: 3 years
- Drug: İbuprofen, Weight: 30kg, Age: 10 years

## Notes

- PDF files in this folder are ignored by git (see `.gitignore`)
- Large PDFs (>50 pages) may take 1-2 minutes to process
- The system works best with well-formatted, text-based PDFs
- Scanned PDFs may require OCR preprocessing

## Data Privacy

**Important:** Only use publicly available medical guidelines for this demo. Do not upload:
- Patient records
- Confidential medical information
- Copyrighted materials without permission
