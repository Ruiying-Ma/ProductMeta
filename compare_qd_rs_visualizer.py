from Visualizer import VisualizerProductMetaSrcGen, VisualizerColorType, Visualizer, VisualizerLineStyleType
from ProductMetaSourceGen import Entry
from Embedder import SentenceTransformerTextEmbedder
from Embedder import AutoTokenizerTextEmbedder
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

from typing import List
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from utils import get_dictionary
import numpy as np
import umap
import torch

# EMBEDDER = AutoTokenizerTextEmbedder()
EMBEDDER = SentenceTransformerTextEmbedder()
# REDUCER = PCA(n_components=2)
# REDUCER = umap.UMAP(n_components=2, random_state=42)
REDUCER = TSNE(n_components=2, random_state=42)

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
    return EMBEDDER.create_embedding(source_list)

def generate_2d_point(embedding_list):
    normalized_entry_embeddings = StandardScaler().fit_transform(embedding_list)
    points = REDUCER.fit_transform(normalized_entry_embeddings)
    return points

def generate_2d_circle_point(embedding_list):
    from Visualizer import VisualizerProductMetaSrcGen
    normalized_embedding_list = StandardScaler().fit_transform(embedding_list)
    if isinstance(REDUCER, PCA):
        REDUCER.fit_transform(normalized_embedding_list)
        base_ax_x = list(REDUCER.components_[0])
        base_ax_y = list(REDUCER.components_[1])

        points = []
        vis_func = VisualizerProductMetaSrcGen()
        for embedding in embedding_list:
            entry = Entry(None, "", None, None, None)
            entry.embedding = embedding
            entry.set_theta(base_ax_x, base_ax_y)
            theta = vis_func._theta_to_radian(entry.theta)
            endpoint = vis_func._circle_to_cartesian(theta, vis_func._quality_to_radius(0.0))
            points.append(endpoint)
    elif isinstance(REDUCER, umap.UMAP) or isinstance(REDUCER, TSNE):
        umap_points = REDUCER.fit_transform(normalized_embedding_list)
        points = []
        for p in umap_points:
            # set_theta
            y = p[1]
            x = p[0]
            phi = np.arctan2(y, x)
            theta = float(np.clip(phi/np.pi, -1, 1))
            # _theta_to_radian
            radian = np.clip(theta*np.pi, -np.pi, np.pi)
            # _circle_to_cartesian
            radius = 1.0
            endpoint = tuple([radius * np.cos(radian), radius * np.sin(radian)])
            points.append(endpoint)
    else:
        raise TypeError(f"Unknown REDUCER type: {type(REDUCER)}")
            

    return np.array(points)

def scatter_embeddings():
    qd_record_jsonl_path = "/home/v-ruiyingma/ProductMeta/log-kettle-20250331-125816/srcgen_result/record.jsonl"
    rs_record_jsonl_path = "/home/v-ruiyingma/ProductMeta/log-kettle-20250331-155135-repeated-sampling-sources-init-qd-20250331-125816/srcgen_result/record.jsonl"
    qd_entries, rs_entries = load_entries(qd_record_jsonl_path, rs_record_jsonl_path)
    
    qd_embedding_list = generate_embedding(qd_entries)
    qd_color_list = [VisualizerColorType.CYAN for _ in qd_embedding_list]
    for i in range(10):
        qd_color_list[i] = VisualizerColorType.BLUE

    rs_embedding_list = generate_embedding(rs_entries)
    rs_color_list = [VisualizerColorType.PINK for _ in rs_embedding_list]
    for i in range(10):
        rs_color_list[i] = VisualizerColorType.MAGENTA

    target_embedding  = [EMBEDDER.create_embedding("kettle")]
    target_color_list = [VisualizerColorType.RED]

    # ### Second test
    qd_embedding_list_2 = []
    qd_color_list_2 = []
    rs_embedding_list_2 = []
    rs_color_list_2 = []

    # qd_record_jsonl_path_2 = "/home/v-ruiyingma/ProductMeta/log-kettle-20250328-231225/srcgen_result/record.jsonl"
    # rs_record_jsonl_path_2 = "/home/v-ruiyingma/ProductMeta/log-kettle-20250331-133748-repeated-sampling-sources/srcgen_result/record.jsonl"
    # qd_entries_2, rs_entries_2 = load_entries(qd_record_jsonl_path_2, rs_record_jsonl_path_2)
    
    # qd_embedding_list_2 = generate_embedding(qd_entries_2)
    # qd_color_list_2 = [VisualizerColorType.YELLOW for _ in qd_embedding_list_2]
    # for i in range(10):
    #     qd_color_list_2[i] = VisualizerColorType.ORANGE

    # rs_embedding_list_2 = generate_embedding(rs_entries_2)
    # rs_color_list_2 = [VisualizerColorType.LIME for _ in rs_embedding_list_2]
    # for i in range(10):
    #     rs_color_list_2[i] = VisualizerColorType.GREEN
    
    # ###

    eng_word_list = [w for w in get_dictionary(size=None) if w.strip().lower() not in STOPWORDS]
    eng_word_embedding_list = EMBEDDER.create_embedding(eng_word_list)
    eng_word_color_list = [VisualizerColorType.LIGHTGREY for _ in eng_word_embedding_list]

    embedding_list = target_embedding + qd_embedding_list + qd_embedding_list_2 + rs_embedding_list + rs_embedding_list_2 + eng_word_embedding_list
    pca_points = generate_2d_point(embedding_list[::-1])
    # pca_points = generate_2d_circle_point(embedding_list[::-1])
    colors = (target_color_list + qd_color_list + qd_color_list_2 + rs_color_list + rs_color_list_2 + eng_word_color_list)[::-1]

    ## filter points
    # new_pca_points = np.array([p for p, c in zip(pca_points, colors) if c not in [VisualizerColorType.LIGHTGREY]])
    # new_colors = [c for c in colors if c not in [VisualizerColorType.LIGHTGREY]]
    # print(len(new_pca_points), len(new_colors))
    # assert len(new_pca_points) == len(new_colors)
    ##

    vis = Visualizer(
        suptitle="QD=[blue, orange], RS=[green, purple], INIT=red, DICT=gray",
        fig_h=5,
        fig_w=5
    )
    import matplotlib.patches as patches
    vis.ax.add_patch(patches.Circle((0, 0), 1.0, edgecolor=VisualizerColorType.WHITE.value, facecolor='none', linewidth=vis.line_width, linestyle=VisualizerLineStyleType.DASHED.value)) # inner circle
    vis.scatter_png(
        points=pca_points,
        colors=colors,
        png_path="compare_qd_rs_tsne_2d_tot_rs_with_init.png",
        need_save=True,
        need_reset=True
    )

def generate_novelty(normalized_embedding_torch: torch.Tensor, k_neighbors):
    if len(normalized_embedding_torch) <= k_neighbors:
        return [0 for _ in normalized_embedding_torch]
    
    similarity = torch.mm(normalized_embedding_torch, normalized_embedding_torch.t())
    distances = 1 - similarity
    # Set self-distance to a high value to exclude from nearest neighbor calculation
    # Use a value that can safely be represented in all precisions
    max_val = torch.finfo(distances.dtype).max / 2
    for i in range(distances.shape[0]):
        distances[i, i] = max_val
    # Get k nearest neighbors for each genome
    sorted_dist, _ = torch.sort(distances, dim=1)
    k_nearest = sorted_dist[:, :k_neighbors]
    # Compute novelty as average distance to k nearest neighbors
    novelty_scores = k_nearest.mean(dim=1)
    assert len(novelty_scores) == len(normalized_embedding_torch)
    return novelty_scores.tolist()
    
def plot_novelty():
    target_embedding  = [EMBEDDER.create_embedding("kettle")]
    target_color_list = [VisualizerColorType.RED]
    
    qd_record_jsonl_path = "/home/v-ruiyingma/ProductMeta/log-kettle-20250331-125816/srcgen_result/record.jsonl"
    rs_record_jsonl_path = "/home/v-ruiyingma/ProductMeta/log-kettle-20250331-155135-repeated-sampling-sources-init-qd-20250331-125816/srcgen_result/record.jsonl"
    qd_entries, rs_entries = load_entries(qd_record_jsonl_path, rs_record_jsonl_path)
    
    qd_embedding_list = generate_embedding(qd_entries)
    qd_color_list = [VisualizerColorType.CYAN for _ in qd_embedding_list]
    for i in range(10):
        qd_color_list[i] = VisualizerColorType.BLUE

    rs_embedding_list = generate_embedding(rs_entries)
    rs_color_list = [VisualizerColorType.PINK for _ in rs_embedding_list]
    for i in range(10):
        rs_color_list[i] = VisualizerColorType.MAGENTA

    # embedding_list = 

    


if __name__ == "__main__":
    plot_novelty()