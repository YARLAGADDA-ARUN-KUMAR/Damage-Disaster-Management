import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import rasterio.features
import shapely.geometry

from src.config import CLASS_MAPPING_5


def majority_vote_damage(
    labeled_mask: np.ndarray,
    damage_probs: np.ndarray,
    min_area: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    K = int(labeled_mask.max())
    if K == 0:
        return np.zeros_like(labeled_mask, dtype=np.uint8), np.array([]), np.array([]), np.array([])

    flat_labels = labeled_mask.ravel()
    counts = np.bincount(flat_labels, minlength=K + 1)
    sums = np.zeros((4, K + 1), dtype=np.float32)

    for c in range(4):
        sums[c] = np.bincount(flat_labels, weights=damage_probs[c].ravel(), minlength=K + 1)

    means = sums[:, 1:] / np.maximum(counts[1:], 1)
    best_classes = (means.argmax(axis=0) + 1).astype(np.uint8)
    best_confs = means.max(axis=0)
    areas = counts[1:].astype(np.int32)

    lookup = np.zeros(K + 1, dtype=np.uint8)
    lookup[1:] = best_classes

    if min_area > 0:
        lookup[counts < min_area] = 0

    voted_mask = lookup[labeled_mask]
    return voted_mask, best_classes, best_confs, areas


def vectorize_predictions(
    loc_mask: np.ndarray,
    damage_probs: np.ndarray,
    min_area: int = 15,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    H, W = loc_mask.shape
    binary_loc = (loc_mask > 0).astype(np.uint8)

    shapes = list(rasterio.features.shapes(binary_loc, mask=(binary_loc > 0)))
    if not shapes:
        return np.zeros((H, W), dtype=np.uint8), []

    labeled_shapes = [(geom, idx + 1) for idx, (geom, _) in enumerate(shapes)]
    labeled_mask = rasterio.features.rasterize(labeled_shapes, out_shape=(H, W), dtype=np.int32)

    voted_mask, best_classes, best_confs, areas = majority_vote_damage(
        labeled_mask, damage_probs, min_area=min_area
    )

    features = []
    for idx, (geom, _) in enumerate(shapes):
        area = int(areas[idx])
        if area < min_area:
            continue

        best_cls = int(best_classes[idx])
        conf = float(best_confs[idx])
        poly_obj = shapely.geometry.shape(geom)
        simplified_poly = poly_obj.simplify(0.5, preserve_topology=True)

        features.append({
            "type": "Feature",
            "id": idx + 1,
            "geometry": shapely.geometry.mapping(simplified_poly),
            "properties": {
                "building_id": idx + 1,
                "area_pixels": area,
                "damage_id": best_cls,
                "damage_class": CLASS_MAPPING_5[best_cls],
                "confidence": round(conf, 4),
            },
        })

    return voted_mask, features


def classify_existing_polygons(
    polygons: List[Union[shapely.geometry.Polygon, Dict[str, Any]]],
    damage_probs: np.ndarray,
    out_shape: Tuple[int, int] = (1024, 1024),
    min_area: int = 15,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if not polygons:
        return np.zeros(out_shape, dtype=np.uint8), []

    labeled_shapes = []
    poly_objects = []

    for idx, p in enumerate(polygons):
        if isinstance(p, dict):
            geom = p
            poly_obj = shapely.geometry.shape(p)
        else:
            geom = shapely.geometry.mapping(p)
            poly_obj = p
        labeled_shapes.append((geom, idx + 1))
        poly_objects.append(poly_obj)

    labeled_mask = rasterio.features.rasterize(labeled_shapes, out_shape=out_shape, dtype=np.int32)
    voted_mask, best_classes, best_confs, areas = majority_vote_damage(
        labeled_mask, damage_probs, min_area=min_area
    )

    features = []
    for idx, poly_obj in enumerate(poly_objects):
        area = int(areas[idx]) if idx < len(areas) else 0
        if area < min_area:
            continue

        best_cls = int(best_classes[idx])
        conf = float(best_confs[idx])

        features.append({
            "type": "Feature",
            "id": idx + 1,
            "geometry": shapely.geometry.mapping(poly_obj),
            "properties": {
                "building_id": idx + 1,
                "area_pixels": area,
                "damage_id": best_cls,
                "damage_class": CLASS_MAPPING_5[best_cls],
                "confidence": round(conf, 4),
            },
        })

    return voted_mask, features


def export_geojson(
    features: List[Dict[str, Any]],
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    geojson_data = {
        "type": "FeatureCollection",
        "metadata": metadata or {},
        "features": features,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)

    return output_path
