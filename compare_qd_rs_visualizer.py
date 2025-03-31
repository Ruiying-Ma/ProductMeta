from Visualizer import VisualizerProductMetaSrcGen, VisualizerColorType, Visualizer
from ProductMetaSourceGen import Entry
from Embedder import SentenceTransformerTextEmbedder
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

from typing import List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import get_dictionary

def load_entries(
        qd_record_jsonl_path: str,
        rs_record_jsonl_path: str
):
    vis = VisualizerProductMetaSrcGen()
    qd_entries = vis.load_entries(
        record_jsonl_path=qd_record_jsonl_path,
        recover_embedding=False
    )
    rs_entries = vis.load_entries(
        record_jsonl_path=rs_record_jsonl_path,
        recover_embedding=False
    )
    return qd_entries, rs_entries

def generate_embedding(entry_list: List[Entry]):
    source_list = [e.source for e in entry_list if e.source != None]
    return SentenceTransformerTextEmbedder().create_embedding(source_list)

def generate_pca_point(embedding_list):
    pca = PCA(n_components=2)
    normalized_entry_embeddings = StandardScaler().fit_transform(embedding_list)
    points = pca.fit_transform(normalized_entry_embeddings)
    return points

if __name__ == "__main__":
    qd_record_jsonl_path = "/home/v-ruiyingma/ProductMeta/log-kettle-20250324-203700/srcgen_result/record.jsonl"
    rs_record_jsonl_path = "/home/v-ruiyingma/ProductMeta/log-kettle-20250327-001131-repeated-sampling-sources/srcgen_result/record.jsonl"
    qd_entries, rs_entries = load_entries(qd_record_jsonl_path, rs_record_jsonl_path)
    
    qd_embedding_list = generate_embedding(qd_entries)
    qd_color_list = [VisualizerColorType.CYAN for _ in qd_embedding_list]
    for i in range(10):
        qd_color_list[i] = VisualizerColorType.BLUE

    rs_embedding_list = generate_embedding(rs_entries)
    rs_color_list = [VisualizerColorType.LIME for _ in rs_embedding_list]
    for i in range(10):
        rs_color_list[i] = VisualizerColorType.GREEN

    target_embedding  = [SentenceTransformerTextEmbedder().create_embedding("kettle")]
    target_color_list = [VisualizerColorType.RED]

    ### Second test
    qd_record_jsonl_path_2 = "/home/v-ruiyingma/ProductMeta/log-kettle-20250328-231225/srcgen_result/record.jsonl"
    rs_record_jsonl_path_2 = "/home/v-ruiyingma/ProductMeta/log-kettle-20250329-000211-repeated-sampling-sources/srcgen_result/record.jsonl"
    qd_entries_2, rs_entries_2 = load_entries(qd_record_jsonl_path_2, rs_record_jsonl_path_2)
    
    qd_embedding_list_2 = generate_embedding(qd_entries_2)
    qd_color_list_2 = [VisualizerColorType.YELLOW for _ in qd_embedding_list_2]
    for i in range(10):
        qd_color_list_2[i] = VisualizerColorType.ORANGE

    rs_embedding_list_2 = generate_embedding(rs_entries_2)
    rs_color_list_2 = [VisualizerColorType.PINK for _ in rs_embedding_list_2]
    for i in range(10):
        rs_color_list_2[i] = VisualizerColorType.MAGENTA
    ###

    eng_word_list = [w for w in get_dictionary(size=None) if w.strip().lower() not in STOPWORDS]
    eng_word_embedding_list = SentenceTransformerTextEmbedder().create_embedding(eng_word_list)
    eng_word_color_list = [VisualizerColorType.LIGHTGREY for _ in eng_word_embedding_list]

    embedding_list = target_embedding + qd_embedding_list + qd_embedding_list_2 + rs_embedding_list + rs_embedding_list_2 + eng_word_embedding_list
    pca_points = generate_pca_point(embedding_list)
    colors = target_color_list + qd_color_list + qd_color_list_2 + rs_color_list + rs_color_list_2 + eng_word_color_list

    ### filter points
    # import numpy as np
    # pca_points_list = list(pca_points)
    # pca_points = np.array([p for p, c in zip(pca_points_list, colors) if c != VisualizerColorType.LIGHTGREY])
    # colors = colors[:len(pca_points)]
    ###

    Visualizer(
        suptitle="QD=[blue, orange], RS=[green, purple], INIT=red, DICT=gray"
    ).scatter_png(
        points=pca_points,
        colors=colors,
        png_path="compare_qd_rs_pca_2d_tot_word_only.png",
        need_save=True,
        need_reset=True
    )