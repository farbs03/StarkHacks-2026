import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backend_bases import PickEvent
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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
    raw: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize captured POIs in 3D.")
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
                raw=data,
            )
        )
    return pois


def make_color_lookup(labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    return {label: cmap(i % 10) for i, label in enumerate(unique_labels)}


def show_viewer(pois: list[POI]) -> None:
    if not pois:
        print("[INFO] No POIs matched your filters.")
        return

    xs = [p.x for p in pois]
    ys = [p.y for p in pois]
    zs = [p.z for p in pois]
    labels = [p.event_type for p in pois]
    colors = make_color_lookup(labels)
    point_colors = [colors[label] for label in labels]

    fig = plt.figure(figsize=(12, 7))
    ax3d = fig.add_subplot(121, projection="3d")
    ax_img = fig.add_subplot(122)
    ax_img.axis("off")

    scatter = ax3d.scatter(xs, ys, zs, c=point_colors, s=45, picker=True)
    ax3d.set_title("POI 3D View (camera-local)")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")

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
    ax3d.legend(handles=handles, title="Event Type", loc="best")

    title_text = ax_img.text(
        0.0,
        0.98,
        "Click a point to preview image and metadata",
        va="top",
        ha="left",
        transform=ax_img.transAxes,
        fontsize=10,
    )

    def on_pick(event: PickEvent) -> None:
        if event.artist is not scatter or len(event.ind) == 0:
            return
        idx = int(event.ind[0])
        poi = pois[idx]

        ax_img.clear()
        img_path = Path(poi.image_path)
        if img_path.exists():
            try:
                image = plt.imread(str(img_path))
                ax_img.imshow(image)
                ax_img.axis("off")
            except Exception:
                ax_img.text(
                    0.0,
                    0.5,
                    f"Could not open image:\n{img_path}",
                    transform=ax_img.transAxes,
                )
                ax_img.axis("off")
        else:
            ax_img.text(
                0.0,
                0.5,
                f"Image not found:\n{img_path}",
                transform=ax_img.transAxes,
            )
            ax_img.axis("off")

        details = (
            f"poi_id={poi.poi_id}\n"
            f"event_type={poi.event_type}\n"
            f"confidence={poi.confidence:.3f}\n"
            f"xyz=({poi.x:.3f}, {poi.y:.3f}, {poi.z:.3f})\n"
            f"description={poi.description[:120]}"
        )
        title_text.set_text(details)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    poi_path = Path(args.poi_jsonl)
    pois = load_pois(
        poi_path,
        min_confidence=args.min_confidence,
        event_type_filter=args.event_type,
    )
    print(f"[INFO] Loaded {len(pois)} POIs from {poi_path}")
    show_viewer(pois)


if __name__ == "__main__":
    main()
