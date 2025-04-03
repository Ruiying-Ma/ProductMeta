import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import scipy.stats
import logging
from enum import Enum
from typing import List, Tuple
logging.disable(logging.DEBUG)
from ProductMetaSourceGen import Entry
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from Embedder import SentenceTransformerTextEmbedder
# EMBEDDER = SentenceTransformerTextEmbedder()
from Embedder import AutoTokenizerTextEmbedder
EMBEDDER = AutoTokenizerTextEmbedder()


class VisualizerColorType(Enum):
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    RED="r"
    GREEN="g"
    BLUE="b"
    YELLOW="y"
    GRAY="gray"
    CYAN="cyan"
    MAGENTA="m"
    ORANGE="orange"
    PINK="pink"
    LIME="lime"
    LIGHTGREY="lightgrey"
    WHITE="white"

class VisualizerFigureType(Enum):
    GIF=0,
    PNG=1,
    BOTH=2,

class VisualizerLineStyleType(Enum):
    SOLID="-"
    DASHED="--"
    DASHDOT="-."
    DOTTED=":"

# Currently only support 2D
class Visualizer:
    def __init__(
            self, 
            fig_w: int=None, 
            fig_h: int=None, 
            x_lim: float=None, 
            y_lim: float=None, 
            x_label: str='X', 
            y_label: str='Y', 
            suptitle: str=None,
            title: str=None,
        ):
        self.fig_w = fig_w
        self.fig_h = fig_h
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.x_label = x_label
        self.y_label = y_label
        self.suptitle = suptitle
        self.title = title
        self.reset()
        
    
    def reset(self):
        if self.fig_w != None and self.fig_h != None:
            self.fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        else:
            self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        if self.x_lim != None:
            self.ax.set_xlim(-self.x_lim, self.x_lim)
        if self.y_lim != None:
            self.ax.set_ylim(-self.y_lim, self.y_lim)
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        if self.suptitle != None:
            self.fig.suptitle(self.suptitle, fontweight="bold")
        if self.title != None:
            self.ax.set_title(self.title)
        w, h = self.fig.get_size_inches()
        
        self.fontsize = w * 1.6
        self.dot_size = w / 2.5
        self.line_width = w / 20


    def scatter_gif(
        self, 
        points: np.ndarray, # 2D points
        colors: List[VisualizerColorType], 
        gif_path: str, 
        frame_per_sec: float=1, 
        points_per_frame=None,
        need_save: bool=True,
        need_reset: bool=True
    ):
        assert gif_path.endswith(".gif")
        logging.info("Dynamically scatter 2D points...")
        # Prepare frame range
        if points_per_frame != None:
            num_frames = int(len(points) / points_per_frame)
            if len(points) % points_per_frame != 0:
                num_frames += 1
            frame_range = [(off * points_per_frame, min(off * points_per_frame + points_per_frame, len(points))) for off in range(num_frames)]
        else:
            frame_range = []
            fid_s = 0
            while fid_s < len(colors):
                fid_e = fid_s + 1
                while fid_e < len(colors):
                    if colors[fid_e] == colors[fid_s]:
                        fid_e += 1
                    else:
                        break
                assert fid_e <= len(points)
                frame_range.append((fid_s, fid_e))
                fid_s = fid_e
        # Prepare animation
        def update(frame):
            fid_s, fid_e = frame_range[frame]
            points_fseq = points[fid_s:fid_e]
            colors_fseq = colors[fid_s:fid_e]
            self.ax.scatter(points_fseq[:, 0], points_fseq[:, 1], c=np.array([c.value for c in colors_fseq]), s=self.dot_size)
            plt.title(f"Point [{fid_s} , {fid_e})") 

        ani = FuncAnimation(self.fig, update, frames=len(frame_range), repeat=True)
        if need_save == True:
            ani.save(gif_path, writer=PillowWriter(fps=frame_per_sec))
            logging.info(f"GIF saved as {gif_path}")
        if need_reset == True:
            self.reset()

    def scatter_png(
        self, 
        points: np.ndarray,
        colors: List[VisualizerColorType],
        png_path: str,
        need_save: bool=True,
        need_reset: bool=True
    ):
        assert png_path.endswith(".png")
        logging.info("Statically scatter 2D points...")
        self.ax.scatter(points[:, 0], points[:, 1], c=np.array([c.value for c in colors]), s=self.dot_size)
        if need_save == True:
            plt.legend()
            plt.savefig(png_path)
            logging.info(f"PNG saved as {png_path}")
        if need_reset == True:
            self.reset()

    def plot_png(
        self, 
        lines: List[Tuple[Tuple, Tuple]], #  list of endpoints ((x1, y1), (x2, y2))
        colors: List[VisualizerColorType], 
        png_path: str,
        need_save: bool=True,
        need_reset: bool=True,
        line_style: VisualizerLineStyleType=VisualizerLineStyleType.SOLID
    ):
        assert png_path.endswith(".png")
        logging.info("Statically plot 2D line segments...")
        for end_points, color in zip(lines, colors):
            x1 = end_points[0][0]
            y1 = end_points[0][1]
            x2 = end_points[1][0]
            y2 = end_points[1][1]
            self.ax.plot([x1, x2], [y1, y2], color=color.value, linewidth=self.line_width, linestyle=line_style.value)
        if need_save == True:
            plt.legend()
            plt.savefig(png_path)
            logging.info(f"PNG saved as {png_path}")
        if need_reset == True:
            self.reset()


class VisualizerProductMetaSrcGen(Visualizer):
    def _theta_to_radian(self, theta: float):
        '''
        - theta: [-1, 1]
        - Return: [-pi, pi]
        '''
        return np.clip(theta*np.pi, -np.pi, np.pi)
    
    def _circle_to_cartesian(self, theta: float, radius: float):
        '''
        - theta: [-pi, pi]

        Return:
        - x = radius cos(theta)
        - y = radius sin(theta)
        '''
        return radius * np.cos(theta), radius * np.sin(theta)
    
    def _category_to_radian(self, category: int, cell_num: int):
        '''
        Return [-pi, pi]
        '''
        return np.clip((category * 2 / cell_num - 1) * np.pi, -np.pi, np.pi)
    
    def _quality_to_radius(self, quality):
        return quality + 1


    def load_entries(
        self, 
        record_jsonl_path: str, 
        recover_embedding: bool
    ) -> List[Entry]:
        entries = []
        with open(record_jsonl_path, 'r') as file:
            for line in file:
                entries.append(Entry.from_jsonl(
                    jsonl=line,
                    recover_embedding=recover_embedding,
                    recover_theta_and_category=False,
                    base_ax_x=None,
                    base_ax_y=None,
                    cell_num=None
                ))
        return entries
    
    
    def archive(
        self,
        cell_num: int,
        record_jsonl_path: str,
        figure_type: VisualizerFigureType,
        figure_path: str,
        frame_per_sec: float=1, 
        points_per_frame = None
    ):
        # Load 2D points and colors
        entries = self.load_entries(
            record_jsonl_path=record_jsonl_path,
            recover_embedding=False
        )

        # Initialize the figure
        qualities = [e.quality for e in entries]
        min_quality = min(qualities)
        max_quality = max(qualities)
        self.ax.add_patch(patches.Circle((0, 0), self._quality_to_radius(min_quality), edgecolor=VisualizerColorType.GRAY.value, facecolor='none', linewidth=self.line_width, linestyle=VisualizerLineStyleType.DASHED.value)) # inner circle
        self.ax.add_patch(patches.Circle((0, 0), self._quality_to_radius(max_quality), edgecolor=VisualizerColorType.GRAY.value, facecolor='none', linewidth=self.line_width, linestyle=VisualizerLineStyleType.DASHED.value)) # inner circle
        if cell_num != None:
            for i in range(cell_num):
                angle = np.pi * 2 * i / cell_num
                self.ax.plot([0, np.cos(angle) * self._quality_to_radius(max_quality)], [0, np.sin(angle) * self._quality_to_radius(max_quality)], color=VisualizerColorType.GRAY.value, linewidth=1)
        
        source_list = [e.source for e in entries]
        assert all([s != None for s in source_list])
        embedding_list = EMBEDDER.create_embedding(source_list)
        normalized_embedding_list = StandardScaler().fit_transform(embedding_list)
        pca = PCA(n_components=2)
        pca.fit_transform(normalized_embedding_list)
        base_ax_x = list(pca.components_[0])
        base_ax_y = list(pca.components_[1])

        points = []
        colors = []
        assert len(embedding_list) == len(entries)
        for entry, embedding in zip(entries, embedding_list):
            assert isinstance(entry, Entry)
            entry.embedding = embedding
            assert entry.embedding != None
            assert entry.quality != None
            entry.set_theta(base_ax_x, base_ax_y)
            theta = self._theta_to_radian(entry.theta)
            endpoint = self._circle_to_cartesian(theta, self._quality_to_radius(entry.quality))
            color = VisualizerColorType.BLUE
            points.append(endpoint)
            colors.append(color)
        
        for i in range(10):
            colors[i] = VisualizerColorType.RED

        points = np.array(points)
        # Visualize
        if figure_type == VisualizerFigureType.GIF or figure_type == VisualizerFigureType.BOTH:
            self.scatter_gif(
                points=points, 
                colors=colors, 
                gif_path=figure_path.replace(".png", ".gif"), 
                frame_per_sec=frame_per_sec, 
                points_per_frame=points_per_frame
            )
        if figure_type == VisualizerFigureType.PNG or figure_type == VisualizerFigureType.BOTH:
            self.scatter_png(
                points=points[::-1], 
                colors=colors[::-1], 
                png_path=figure_path.replace(".gif", ".png")
            )

    def pca_2d(
        self,
        record_jsonl_path: str,
        figure_path: str,
        figure_type: VisualizerFigureType,
        frame_per_sec: float=1, 
        points_per_frame = None
    ):
        # Load 2D points and colors
        entries = self.load_entries(
            record_jsonl_path=record_jsonl_path,
            recover_embedding=False
        )

        colors = []
        sources = []
        for entry in entries:
            assert isinstance(entry, Entry)
            assert entry.source != None
            sources.append(entry.source)
            colors.append(VisualizerColorType.BLUE)

        for i in range(10):
            colors[i] = VisualizerColorType.RED
            
        entry_embeddings = EMBEDDER.create_embedding(sources)
        pca = PCA(n_components=2)
        normalized_entry_embeddings = StandardScaler().fit_transform(entry_embeddings)
        points = pca.fit_transform(normalized_entry_embeddings)

        if figure_type == VisualizerFigureType.GIF or figure_type == VisualizerFigureType.BOTH:
            self.scatter_gif(
                points=points, 
                colors=colors, 
                gif_path=figure_path.replace(".png", ".gif"), 
                frame_per_sec=frame_per_sec, 
                points_per_frame=points_per_frame
            )
        if figure_type == VisualizerFigureType.PNG or figure_type == VisualizerFigureType.BOTH:
            self.scatter_png(
                points=points[::-1], 
                colors=colors[::-1], 
                png_path=figure_path.replace(".gif", ".png")
            )

if __name__ == "__main__":
    import json

    visualizer = VisualizerProductMetaSrcGen(
        suptitle="ProductMetaSrcGen",
        fig_w=10,
        fig_h=10,
    )

    record_jsonl_path = "/home/v-ruiyingma/ProductMeta/log-kettle-20250331-155135-repeated-sampling-sources-init-qd-20250331-125816/srcgen_result/record.jsonl"
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(record_jsonl_path)), "figure"), exist_ok=True)
    stats_json_path = record_jsonl_path.replace("record.jsonl", "statistics.json")
    with open(stats_json_path, 'r') as file:
        stats = json.load(file)
    
    visualizer.archive( # (w, h) =（10, 10)
        cell_num=100,
        record_jsonl_path=record_jsonl_path,
        figure_type=VisualizerFigureType.BOTH,
        figure_path=os.path.join(os.path.dirname(os.path.dirname(record_jsonl_path)), "figure", "archive_eb5smallv2.png"),
        frame_per_sec=1,
        points_per_frame=stats["update_interval"]# this is your update interval
    )

    visualizer.reset()
    
    visualizer.pca_2d(
        record_jsonl_path=record_jsonl_path,
        figure_path=os.path.join(os.path.dirname(os.path.dirname(record_jsonl_path)), "figure", "pca_2d_eb5smallv2.png"),
        figure_type=VisualizerFigureType.BOTH,
        frame_per_sec=1,
        points_per_frame=stats["update_interval"],
    )
    
