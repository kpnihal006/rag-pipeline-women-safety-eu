# Extraction Report

## Corpus Statistics

| Metric | Value |
|---|---|
| Documents (PDFs) | 31 |
| Total pages | 2270 |
| Total characters | 6933770 |
| Avg characters per page | 3054.5 |
| Near-empty pages (< 50 chars) | 57 |

## Near-Empty Pages Flagged

The following pages had fewer than 50 characters after cleaning:

- `2024 Report on Gender Equality in the EU_coming soon.pdf` — page 1 (0 chars)
- `2024 Report on Gender Equality in the EU_coming soon.pdf` — page 84 (0 chars)
- `48802978de.pdf` — page 2 (0 chars)
- `KS-01-24-013-EN-N.pdf` — page 1 (43 chars)
- `KS-01-24-013-EN-N.pdf` — page 12 (0 chars)
- `KS-01-24-013-EN-N.pdf` — page 21 (45 chars)
- `KS-01-24-013-EN-N.pdf` — page 28 (0 chars)
- `KS-01-24-013-EN-N.pdf` — page 36 (0 chars)
- `KS-01-24-013-EN-N.pdf` — page 37 (27 chars)
- `KS-01-24-013-EN-N.pdf` — page 43 (18 chars)
- `KS-01-24-013-EN-N.pdf` — page 47 (0 chars)
- `KS-01-24-013-EN-N.pdf` — page 48 (0 chars)
- `annual_report_GE_2022_printable_EN.pdf` — page 72 (0 chars)
- `annual_report_GE_2023_web_EN.pdf` — page 78 (0 chars)
- `annual_report_GE_2023_web_EN.pdf` — page 80 (0 chars)
- `beijing30-action-agenda-for-all-women-and-girls-en.pdf` — page 1 (49 chars)
- `care-survey-second-wave.pdf` — page 2 (0 chars)
- `care-survey-second-wave.pdf` — page 134 (0 chars)
- `care-survey-second-wave.pdf` — page 136 (14 chars)
- `eu-gbv-annexes_en.pdf` — page 1 (34 chars)
- `eu-gbv-annexes_en.pdf` — page 20 (43 chars)
- `eu-gender-based-violence-survey-evidence-for-policy-and-practice_4.pdf` — page 26 (0 chars)
- `eu-gender-based-violence-survey-evidence-for-policy-and-practice_4.pdf` — page 42 (0 chars)
- `eu-gender-based-violence-survey-evidence-for-policy-and-practice_4.pdf` — page 43 (46 chars)
- `eu-gender-based-violence-survey-evidence-for-policy-and-practice_4.pdf` — page 58 (0 chars)
- `eu-gender-based-violence-survey-evidence-for-policy-and-practice_4.pdf` — page 68 (0 chars)
- `eu-gender-based-violence-survey-evidence-for-policy-and-practice_4.pdf` — page 87 (19 chars)
- `eu-gender-based-violence-survey-evidence-for-policy-and-practice_4.pdf` — page 99 (10 chars)
- `eu-gender-based-violence-survey-evidence-for-policy-and-practice_4.pdf` — page 109 (23 chars)
- `eu-gender-based-violence-survey-evidence-for-policy-and-practice_4.pdf` — page 126 (0 chars)
- `eu-wide-guidelines-on-gender-neutral-job-evaluation-and-classification-step-by-step-toolkit.pdf` — page 1 (0 chars)
- `eu-wide-guidelines-on-gender-neutral-job-evaluation-and-classification-step-by-step-toolkit.pdf` — page 152 (0 chars)
- `fra-2014-vaw-survey-main-results-apr14_en.pdf` — page 4 (0 chars)
- `fra-2014-vaw-survey-main-results-apr14_en.pdf` — page 56 (0 chars)
- `fra-2014-vaw-survey-main-results-apr14_en.pdf` — page 96 (0 chars)
- `fra-2014-vaw-survey-main-results-apr14_en.pdf` — page 122 (0 chars)
- `fra-2014-vaw-survey-main-results-apr14_en.pdf` — page 140 (0 chars)
- `fra-2014-vaw-survey-main-results-apr14_en.pdf` — page 168 (0 chars)
- `fra-2014-vaw-survey-main-results-apr14_en.pdf` — page 196 (0 chars)
- `fra-2014-vaw-survey-main-results-apr14_en.pdf` — page 197 (0 chars)
- `fra-2014-vaw-survey-main-results-apr14_en.pdf` — page 198 (0 chars)
- `gender-equality-index-2024-tackling-violence-against-women-tackling-gender-inequalities.pdf` — page 91 (32 chars)
- `gender-equality-index-2025-sharper-data-for-a-changing-world (1).pdf` — page 2 (0 chars)
- `gender-equality-index-2025-sharper-data-for-a-changing-world (1).pdf` — page 176 (0 chars)
- `gender-equality-index-2025-sharper-data-for-a-changing-world (1).pdf` — page 178 (14 chars)
- `gender-equality-index-2025-sharper-data-for-a-changing-world.pdf` — page 2 (0 chars)
- `gender-equality-index-2025-sharper-data-for-a-changing-world.pdf` — page 176 (0 chars)
- `gender-equality-index-2025-sharper-data-for-a-changing-world.pdf` — page 178 (14 chars)
- `gender-equality-index-methodological-report.pdf` — page 2 (0 chars)
- `gender-equality-index-methodological-report.pdf` — page 3 (44 chars)
- `gender-equality-index-methodological-report.pdf` — page 78 (14 chars)
- `gender-equality-strategy-2026-2030.pdf` — page 1 (0 chars)
- `sharing-care-closign-gender-gaps-care-survey-2024.pdf` — page 2 (0 chars)
- `sharing-care-closign-gender-gaps-care-survey-2024.pdf` — page 3 (46 chars)
- `sharing-care-closign-gender-gaps-care-survey-2024.pdf` — page 108 (14 chars)
- `wcms_711570.pdf` — page 20 (0 chars)
- `wmid_methodology_31032026.pdf` — page 208 (3 chars)

## Text Cleaning Applied

- Repeated headers/footers removed (lines appearing on 10+ pages)
- Hyphenated line breaks rejoined (e.g. "impor-\ntant" → "important")
- Mid-sentence line breaks merged into single space
- Multiple blank lines collapsed into one
- Multiple spaces collapsed into one

## Configuration

- Library used: PyMuPDF (fitz)
- Input directory: `data/pdfs`
- Output directory: `data`
- Near-empty threshold: 50 characters
- Header/footer repeat threshold: 10 pages
