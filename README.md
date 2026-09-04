# UrbanSS

Requirements: `pip install -r requirements.txt`
Data layout follows `data/{city}_example/` . All regions are defined by `{city}_si_list.csv` (columns: `satellite_img_name`, `WGS84_lower_left`, `WGS84_upper_right`; row index = region id). Every embedding/label file is aligned to it.

## 1. POI Semantic Module
Input: `{city}_POI_list.csv` → generate two prompts (`cat_prompt`, `surrounding_prompt`) with the method in the paper → `{city}_POI_With_prompts.csv` (`region_id` maps each POI to a region in `{city}_si_list.csv`)
Scripts (the two encoders are alternatives):

- `poi/get_embedding_llm_encoder.py` or `poi/get_embedding_sentence_encoder.py`
- then `poi/gen_region_e.py`
- then `poi/train.py`
Output: POI embedding

## 2. Satellite Image Module
Input: `{city}_si_list.csv` (satellite image names) + `satellite_imagery/` (the image files) + positives index file (`img/construct_positive.py`) + height crops from Module 3 (the height statistics used for hard negatives are computed from them inside `img/train.py`)
Scripts (run from `img/`): `img/train.py` → `img/feature_extract.py`
Output: image embedding

## 3. Building Height Module
Input: `{city}_si_list.csv` (WGS84 bounds) + building-height raster (tif)
Script: `height.py`
Output: height embedding + height crops (used by Module 2)

## 4. Downstream Prediction
Input: POI / image / height embeddings from the steps above + `{city}_{task}.csv`
Script: `mlp_AOSF.py`
