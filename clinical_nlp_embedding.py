import pandas as pd
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("WARNING: Chưa cài đặt sentence-transformers. Hãy chạy lệnh: pip install sentence-transformers")

def df_to_text_prompts(df_clinical_labs):
    """
    Prompt v1 — mô tả đầy đủ, dùng cho all-MiniLM-L6-v2.
    """
    prompts = []
    df_clean = df_clinical_labs.fillna(0)

    for idx, row in df_clean.iterrows():
        age = int(row.get('age', 0))
        pack_years = row.get('pack_years', 0)
        ecog = row.get('ecog', 0)
        albumin = row.get('albumin', 0)
        dnlr = row.get('dnlr', 0)
        tumor_burden = row.get('tumor_burden', 0)

        brain_mets = "with" if row.get('brain_mets') == 1 else "without"
        liver_mets = "with" if row.get('liver_mets') == 1 else "without"
        therapy_line = int(row.get('therapy_line', 1))
        combo_therapy = "combination therapy" if row.get('recieves_combo_therapy') == 1 else "monotherapy"
        site_lung = "lung" if row.get('site_lung') == 1 else "extra-pulmonary"
        pdl1 = "received" if row.get('recieves_pdl1_therapy') == 1 else "did not receive"
        hist = "adenocarcinoma" if row.get('hist_adeno') == 1 else "non-adenocarcinoma"

        prompt = (f"The patient is a {age}-year-old. Smoking history in pack-years is {pack_years}. "
                 f"ECOG performance status is {ecog}. Blood albumin concentration is {albumin:.2f}. "
                 f"Derived neutrophil-to-lymphocyte ratio (dNLR) is {dnlr:.2f}. Initial tumor burden stands at {tumor_burden:.2f}. "
                 f"The patient is diagnosed with {hist} at the {site_lung} site. "
                 f"Metastatic status: {brain_mets} brain metastasis and {liver_mets} liver metastasis. "
                 f"Currently undergoing therapy line {therapy_line}, utilizing {combo_therapy}. "
                 f"The patient {pdl1} prior PD-L1 immunotherapy.")

        prompts.append(prompt)

    return prompts


def df_to_text_prompts_v2(df_clinical_labs):
    """
    Prompt v2 — nhấn mạnh yếu tố tiên lượng ICI, phù hợp với Bio_ClinicalBERT.
    Ngắn hơn v1, tập trung vào risk factors lâm sàng liên quan đến đáp ứng miễn dịch.
    """
    prompts = []
    df_clean = df_clinical_labs.fillna(0)

    for idx, row in df_clean.iterrows():
        age = int(row.get('age', 0))
        ecog = row.get('ecog', 0)
        albumin = row.get('albumin', 0)
        dnlr = row.get('dnlr', 0)
        therapy_line = int(row.get('therapy_line', 1))
        brain_mets = "with" if row.get('brain_mets') == 1 else "without"
        liver_mets = "with" if row.get('liver_mets') == 1 else "without"
        combo = "combination" if row.get('recieves_combo_therapy') == 1 else "single-agent"
        hist = "adenocarcinoma" if row.get('hist_adeno') == 1 else "non-adenocarcinoma"

        risk_factors = []
        if ecog >= 2:
            risk_factors.append("poor ECOG performance status")
        if albumin < 3.5:
            risk_factors.append("hypoalbuminemia")
        if dnlr > 3:
            risk_factors.append("elevated dNLR indicating systemic inflammation")
        if row.get('brain_mets') == 1:
            risk_factors.append("brain metastases")
        if row.get('liver_mets') == 1:
            risk_factors.append("liver metastases")
        risk_str = "; ".join(risk_factors) if risk_factors else "no major adverse risk factors"

        prompt = (f"NSCLC {hist} patient receiving line {therapy_line} {combo} ICI immunotherapy. "
                 f"Age {age}, ECOG {int(ecog)}, albumin {albumin:.1f} g/dL, dNLR {dnlr:.1f}. "
                 f"Metastatic status: {brain_mets} brain and {liver_mets} liver involvement. "
                 f"Clinical risk profile: {risk_str}.")

        prompts.append(prompt)

    return prompts


class ClinicalTextEmbedder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Sentence-transformers embedder. all-MiniLM-L6-v2 → 384 dim.
        """
        print(f"   [NLP Module] Loading sentence-transformer: {model_name}...")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Lỗi tải mô hình NLP: {e}")
            raise e
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"   [NLP Module] Load thành công. Dimensionality: {self.dim}")

    def embed_dataframe(self, df_clinical_labs):
        print(f"   [NLP Module] Encoding {len(df_clinical_labs)} prompts (dim={self.dim})...")
        prompts = df_to_text_prompts(df_clinical_labs)

        print("\n   --- VÍ DỤ PROMPT (all-MiniLM) ---")
        print(f"   ▶ {prompts[0]}\n")

        embeddings = self.model.encode(prompts, show_progress_bar=False)
        df_embeddings = pd.DataFrame(embeddings, index=df_clinical_labs.index)
        df_embeddings.columns = [f"nlp_{c}" for c in df_embeddings.columns]
        return df_embeddings


class HFBERTEmbedder:
    """
    Embedder cho các BERT-based model từ HuggingFace (Bio_ClinicalBERT, BiomedBERT).
    Dùng mean pooling trên last hidden state → 768-dim vector.
    Yêu cầu: pip install transformers
    """

    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT'):
        print(f"   [NLP Module] Loading HuggingFace model: {model_name}...")
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.hf_model = AutoModel.from_pretrained(model_name)
            self.hf_model.eval()
            self.torch = torch
        except ImportError:
            raise ImportError("Cần cài thêm: pip install transformers")
        except Exception as e:
            print(f"Lỗi tải model: {e}")
            raise
        self.dim = self.hf_model.config.hidden_size
        self.model_name = model_name
        print(f"   [NLP Module] Load thành công. Embedding dim: {self.dim}")

    def _mean_pool(self, model_output, attention_mask):
        token_emb = model_output.last_hidden_state
        mask_exp = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        return (token_emb * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)

    def embed_dataframe(self, df_clinical_labs, batch_size=32, prompt_version='v2'):
        if prompt_version == 'v2':
            prompts = df_to_text_prompts_v2(df_clinical_labs)
        else:
            prompts = df_to_text_prompts(df_clinical_labs)

        print(f"\n   --- VÍ DỤ PROMPT (Bio_ClinicalBERT v{prompt_version[-1]}) ---")
        print(f"   ▶ {prompts[0]}\n")
        print(f"   [NLP Module] Encoding {len(prompts)} prompts (dim={self.dim})...")

        torch = self.torch
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                encoded = self.tokenizer(batch, padding=True, truncation=True,
                                         max_length=256, return_tensors='pt')
                output = self.hf_model(**encoded)
                emb = self._mean_pool(output, encoded['attention_mask'])
                all_embeddings.append(emb.cpu().numpy())

        embeddings = np.vstack(all_embeddings)
        df_emb = pd.DataFrame(embeddings, index=df_clinical_labs.index)
        df_emb.columns = [f"nlp_{c}" for c in df_emb.columns]
        return df_emb


def prepare_nlp_clinical_modality(df_dict, df_labs, modality_mask, name='cnl_nlp_embedding'):
    """
    all-MiniLM-L6-v2 → 384-dim. Dùng cho các cell cũ.
    """
    embedder = ClinicalTextEmbedder(model_name='sentence-transformers/all-MiniLM-L6-v2')
    df_embeddings = embedder.embed_dataframe(df_labs)

    if name not in modality_mask.columns:
        modality_mask[name] = False
    modality_mask.loc[df_embeddings.index, name] = True
    df_dict[name] = df_embeddings

    print(f"   [NLP Module] Added modality '{name}' shape: {df_embeddings.shape}")
    return df_embeddings


def prepare_clinical_bert_modality(df_dict, df_labs, modality_mask,
                                    name='cnl_bioclinical_embedding',
                                    model_name='emilyalsentzer/Bio_ClinicalBERT',
                                    prompt_version='v2'):
    """
    Bio_ClinicalBERT (hoặc BiomedBERT) → 768-dim.
    Dùng no_scale khi đưa vào DyAM (embeddings đã chuẩn hóa).
    """
    embedder = HFBERTEmbedder(model_name=model_name)
    df_embeddings = embedder.embed_dataframe(df_labs, prompt_version=prompt_version)

    if name not in modality_mask.columns:
        modality_mask[name] = False
    modality_mask.loc[df_embeddings.index, name] = True
    df_dict[name] = df_embeddings

    print(f"   [NLP Module] Added modality '{name}' shape: {df_embeddings.shape}")
    return df_embeddings
