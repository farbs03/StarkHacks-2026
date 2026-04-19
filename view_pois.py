import argparse
import json
import re
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from mpl_toolkits.mplot3d import proj3d


@dataclass(slots=True)
class POI:
    poi_id: str
    event_type: str
    confidence: float
    x: float
    y: float
    z: float
    image_path: str
    description: str
    metadata_path: str
    raw: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI viewer for captured POIs.")
    parser.add_argument(
        "--poi-jsonl",
        default="captures/poi.jsonl",
        help="Path to POI JSONL file exported by pipeline/poi_exporter.py",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Filter out points with lower confidence",
    )
    parser.add_argument(
        "--event-type",
        default="",
        help="Optional event type filter (substring match)",
    )
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_description_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("description", "")).strip()
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    candidates = [text]
    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                extracted = str(obj.get("description", "")).strip()
                if extracted:
                    return extracted
        except json.JSONDecodeError:
            pass

        for idx, ch in enumerate(candidate):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(candidate[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                extracted = str(obj.get("description", "")).strip()
                if extracted:
                    return extracted

    return text


def load_pois(path: Path, min_confidence: float, event_type_filter: str) -> list[POI]:
    if not path.exists():
        raise FileNotFoundError(f"POI JSONL not found: {path}")

    pois: list[POI] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = str(data.get("event_type", "unknown"))
        confidence = _safe_float(data.get("confidence", 0.0), 0.0)
        if confidence < min_confidence:
            continue
        if event_type_filter and event_type_filter.lower() not in event_type.lower():
            continue

        pose = data.get("world_or_local_pose", {})
        xyz = pose.get("local_xyz_m", {})
        x = _safe_float(xyz.get("x", 0.0), 0.0)
        y = _safe_float(xyz.get("y", 0.0), 0.0)
        z = _safe_float(xyz.get("z", 0.0), 0.0)

        pois.append(
            POI(
                poi_id=str(data.get("poi_id", "")),
                event_type=event_type,
                confidence=confidence,
                x=x,
                y=y,
                z=z,
                image_path=str(data.get("image_path", "")),
                description=str(data.get("description", "")),
                metadata_path=str(data.get("metadata_path", "")),
                raw=data,
            )
        )
    return pois


def make_color_lookup(labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    return {label: cmap(i % 10) for i, label in enumerate(unique_labels)}


class POIViewerApp:
    def __init__(self, root: tk.Tk, pois: list[POI]) -> None:
        self.root = root
        self.pois = pois

        self.root.title("POI Viewer - 3D + Event Details")
        self.root.geometry("1500x900")

        self.status_var = tk.StringVar(value="Click a point in the 3D view.")
        self.summary_var = tk.StringVar(value="")
        self._points_xyz = np.empty((0, 3), dtype=float)

        self._build_layout()
        self._build_scatter_plot()
        self._init_detail_views()

        if self.pois:
            self._select_poi(0)

    def _build_layout(self) -> None:
        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        self.left_frame = ttk.Frame(paned)
        self.right_frame = ttk.Frame(paned)
        paned.add(self.left_frame, weight=3)
        paned.add(self.right_frame, weight=2)

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(fill=tk.X, padx=8, pady=4)

        self.scatter_fig = Figure(figsize=(8, 7), dpi=100)
        self.scatter_ax = self.scatter_fig.add_subplot(111, projection="3d")
        self.scatter_canvas = FigureCanvasTkAgg(self.scatter_fig, master=self.left_frame)
        self.scatter_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.scatter_canvas, self.left_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)

        ttk.Label(self.right_frame, text="POIs:").pack(anchor="w", padx=10, pady=(10, 2))
        self.poi_list = tk.Listbox(self.right_frame, height=8, exportselection=False)
        self.poi_list.pack(fill=tk.X, padx=10, pady=(0, 6))
        self.poi_list.bind("<<ListboxSelect>>", self._on_list_select)

        summary = ttk.Label(
            self.right_frame,
            textvariable=self.summary_var,
            justify=tk.LEFT,
            anchor="w",
            wraplength=520,
        )
        summary.pack(fill=tk.X, padx=10, pady=(10, 6))

        self.detail_fig = Figure(figsize=(6, 6), dpi=100)
        self.image_ax = self.detail_fig.add_subplot(211)
        self.sensor_ax = self.detail_fig.add_subplot(212)
        self.detail_canvas = FigureCanvasTkAgg(self.detail_fig, master=self.right_frame)
        self.detail_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        ttk.Label(self.right_frame, text="Natural Language Description:").pack(
            anchor="w", padx=10, pady=(2, 2)
        )
        self.description_text = tk.Text(self.right_frame, height=7, wrap=tk.WORD)
        self.description_text.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.description_text.configure(state=tk.DISABLED)

        for i, poi in enumerate(self.pois):
            self.poi_list.insert(
                tk.END, f"{i + 1}. {poi.event_type} ({poi.confidence:.2f})"
            )

    def _build_scatter_plot(self) -> None:
        if not self.pois:
            self.scatter_ax.set_title("No POIs found")
            self.scatter_canvas.draw_idle()
            return

        xs = [p.x for p in self.pois]
        ys = [p.y for p in self.pois]
        zs = [p.z for p in self.pois]
        self._points_xyz = np.column_stack([xs, ys, zs]).astype(float)
        labels = [p.event_type for p in self.pois]
        colors = make_color_lookup(labels)
        point_colors = [colors[label] for label in labels]

        self.scatter = self.scatter_ax.scatter(xs, ys, zs, c=point_colors, s=48, picker=8)
        self.scatter_ax.set_title("POI 3D View (camera-local)")
        self.scatter_ax.set_xlabel("X (m)")
        self.scatter_ax.set_ylabel("Y (m)")
        self.scatter_ax.set_zlabel("Z (m)")

        handles = []
        for label, color in colors.items():
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=8,
                    label=label,
                )
            )
        self.scatter_ax.legend(handles=handles, title="Event Type", loc="best")
        self.scatter_canvas.mpl_connect("pick_event", self._on_pick)
        self.scatter_canvas.mpl_connect("button_press_event", self._on_click_fallback)
        self.scatter_canvas.draw_idle()

    def _init_detail_views(self) -> None:
        self.image_ax.axis("off")
        self.image_ax.set_title("Event Image")
        self.sensor_ax.set_title("Sensor Snapshot")
        self.sensor_ax.text(0.5, 0.5, "No point selected", ha="center", va="center")
        self.sensor_ax.set_xticks([])
        self.sensor_ax.set_yticks([])
        self.detail_canvas.draw_idle()

    def _on_pick(self, event) -> None:
        if event.artist is not self.scatter or len(event.ind) == 0:
            return
        idx = int(event.ind[0])
        self._select_poi(idx)

    def _on_click_fallback(self, event) -> None:
        # 3D pick events can be unreliable in embedded backends; select nearest point in screen-space.
        if event.inaxes is not self.scatter_ax:
            return
        if event.x is None or event.y is None:
            return
        if self._points_xyz.size == 0:
            return

        projected = np.array(
            [
                proj3d.proj_transform(x, y, z, self.scatter_ax.get_proj())[:2]
                for x, y, z in self._points_xyz
            ]
        )
        screen_xy = self.scatter_ax.transData.transform(projected)
        click = np.array([event.x, event.y], dtype=float)
        distances = np.linalg.norm(screen_xy - click, axis=1)
        idx = int(np.argmin(distances))
        self._select_poi(idx)

    def _on_list_select(self, _event) -> None:
        selection = self.poi_list.curselection()
        if not selection:
            return
        self._select_poi(int(selection[0]))

    def _load_metadata(self, poi: POI) -> dict[str, Any]:
        metadata_path = Path(poi.metadata_path)
        if metadata_path.exists():
            try:
                return json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _select_poi(self, idx: int) -> None:
        poi = self.pois[idx]
        metadata = self._load_metadata(poi)
        if self.poi_list.size() > idx:
            self.poi_list.selection_clear(0, tk.END)
            self.poi_list.selection_set(idx)
            self.poi_list.activate(idx)

        self.summary_var.set(
            "\n".join(
                [
                    f"poi_id: {poi.poi_id}",
                    f"event_type: {poi.event_type}",
                    f"confidence: {poi.confidence:.3f}",
                    f"xyz(m): ({poi.x:.3f}, {poi.y:.3f}, {poi.z:.3f})",
                ]
            )
        )
        self.status_var.set(f"Selected POI {idx + 1}/{len(self.pois)}: {poi.event_type}")

        self._render_image(poi.image_path)
        self._render_sensor_graph(metadata.get("sensor_snapshot"))
        self._render_description(poi.description, metadata)
        self.detail_canvas.draw_idle()

    def _render_image(self, image_path: str) -> None:
        self.image_ax.clear()
        self.image_ax.set_title("Event Image")
        path = Path(image_path)
        if path.exists():
            try:
                img = plt.imread(str(path))
                self.image_ax.imshow(img)
                self.image_ax.axis("off")
                return
            except Exception:
                pass
        self.image_ax.text(0.5, 0.5, f"Image unavailable\n{path}", ha="center", va="center")
        self.image_ax.axis("off")

    def _render_sensor_graph(self, snapshot: Any) -> None:
        self.sensor_ax.clear()
        self.sensor_ax.set_title("Sensor Snapshot")

        if not isinstance(snapshot, dict):
            self.sensor_ax.text(
                0.5,
                0.5,
                "No sensor snapshot in metadata",
                ha="center",
                va="center",
            )
            self.sensor_ax.set_xticks([])
            self.sensor_ax.set_yticks([])
            return

        keys = ["ldr", "ir", "sound", "gas", "dist", "risk"]
        values = [_safe_float(snapshot.get(k, 0.0), 0.0) for k in keys]

        bars = self.sensor_ax.bar(keys, values, color="#4C72B0")
        for bar, val in zip(bars, values):
            self.sensor_ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        self.sensor_ax.set_ylabel("Value")
        self.sensor_ax.grid(axis="y", alpha=0.25)
        self.sensor_ax.tick_params(axis="x", rotation=20)

    def _render_description(self, description: str, metadata: dict[str, Any]) -> None:
        text = _extract_description_text(description)
        if not text:
            text = _extract_description_text(metadata.get("description", ""))
        if not text:
            text = "No natural language description available."

        self.description_text.configure(state=tk.NORMAL)
        self.description_text.delete("1.0", tk.END)
        self.description_text.insert(tk.END, text)
        self.description_text.configure(state=tk.DISABLED)


def main() -> None:
    args = parse_args()
    poi_path = Path(args.poi_jsonl)
    pois = load_pois(
        poi_path,
        min_confidence=args.min_confidence,
        event_type_filter=args.event_type,
    )
    print(f"[INFO] Loaded {len(pois)} POIs from {poi_path}")
    root = tk.Tk()
    POIViewerApp(root, pois)
    root.mainloop()


if __name__ == "__main__":
    main()
