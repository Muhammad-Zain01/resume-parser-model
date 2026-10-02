# Resume Parser Dataset

This project prepares a resume-parsing dataset through four stages:

- `.data/raw/` — source resume files, organized by job category.
- `.data/extracted/` — text extracted from the source files.
- `.data/annotated/` — schema-validated target JSON for resumes.
- `.data/curated/` — train, validation, and test JSONL splits.

## Local setup

From the project root, create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `HF_DATASET_REPO` to the Hugging Face dataset ID, for example `md-zain/resume-parser-dataset`. A Hugging Face token is optional for downloading a public dataset; it is required to upload changes.

Download or update the dataset locally with:

```bash
python download_resume_parser_dataset.py
```

The script downloads the configured dataset into `.data/` and displays a progress bar. It updates files from the remote repository without deleting local-only files.

To upload the complete `.data/` pipeline to the configured public Hugging Face dataset repository, run:

```bash
python upload_resume_parser_dataset.py
```

The uploader reports the file count and size, then uses Hugging Face's Xet transfer progress display. It requires a valid `HF_TOKEN` with write access.

## Dataset sources

- [Mehyaar – Annotated NER PDF Resumes](https://huggingface.co/datasets/Mehyaar/Annotated_NER_PDF_Resumes)
- [opensporks – Resumes](https://huggingface.co/datasets/opensporks/resumes)
- [hadikp – Resume Dataset PDF](https://www.kaggle.com/datasets/hadikp/resume-data-pdf/data)
