from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
}


@dataclass
class ConceptMediaRecord:
    path: str
    prompt: str
    kind: str
    media_mode: str
    video_id: str | None = None
    frame_index: int | None = None

    def to_json(self) -> dict:
        return asdict(self)


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _iter_image_paths(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(path for path in root.rglob("*") if _is_image_file(path))
    return sorted(path for path in root.iterdir() if _is_image_file(path))


def _build_image_records(root: Path, prompt: str, kind: str, recursive: bool) -> list[ConceptMediaRecord]:
    return [
        ConceptMediaRecord(
            path=str(path),
            prompt=prompt,
            kind=kind,
            media_mode="image",
        )
        for path in _iter_image_paths(root, recursive=recursive)
    ]


def _build_video_frame_records(
    root: Path,
    prompt: str,
    kind: str,
    recursive: bool,
) -> list[ConceptMediaRecord]:
    records: list[ConceptMediaRecord] = []
    clip_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for clip_dir in clip_dirs:
        frame_paths = _iter_image_paths(clip_dir, recursive=recursive)
        for frame_index, frame_path in enumerate(frame_paths):
            records.append(
                ConceptMediaRecord(
                    path=str(frame_path),
                    prompt=prompt,
                    kind=kind,
                    media_mode="video_frames",
                    video_id=clip_dir.name,
                    frame_index=frame_index,
                )
            )
    return records


def build_media_records(
    root_dir: str,
    prompt: str,
    kind: str,
    media_mode: str = "image",
    recursive: bool = True,
) -> list[ConceptMediaRecord]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Media directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory of media files: {root}")

    if media_mode == "image":
        records = _build_image_records(root, prompt=prompt, kind=kind, recursive=recursive)
    elif media_mode == "video_frames":
        records = _build_video_frame_records(root, prompt=prompt, kind=kind, recursive=recursive)
    else:
        raise ValueError("media_mode must be 'image' or 'video_frames'")

    if not records:
        raise ValueError(f"No media files found in {root} for media_mode='{media_mode}'")
    return records


class PersonalizationDataset(Dataset):
    def __init__(
        self,
        records: list[ConceptMediaRecord],
        image_transform: Callable | None = None,
    ):
        self.records = records
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
            if self.image_transform is not None:
                image = self.image_transform(image)

        return {
            "image": image,
            "prompt": record.prompt,
            "kind": record.kind,
            "path": record.path,
            "media_mode": record.media_mode,
            "video_id": record.video_id,
            "frame_index": record.frame_index,
        }


def _collate_batch(batch: list[dict]) -> dict:
    return {
        "images": [item["image"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "kinds": [item["kind"] for item in batch],
        "paths": [item["path"] for item in batch],
        "media_modes": [item["media_mode"] for item in batch],
        "video_ids": [item["video_id"] for item in batch],
        "frame_indices": [item["frame_index"] for item in batch],
    }


def build_dataloaders(
    instance_dir: str,
    instance_prompt: str,
    class_dir: str | None = None,
    class_prompt: str | None = None,
    media_mode: str = "image",
    batch_size: int = 4,
    recursive: bool = True,
    image_transform: Callable | None = None,
    num_workers: int = 0,
) -> dict:
    instance_records = build_media_records(
        instance_dir,
        prompt=instance_prompt,
        kind="instance",
        media_mode=media_mode,
        recursive=recursive,
    )
    payload = {
        "instance_records": instance_records,
        "instance_loader": DataLoader(
            PersonalizationDataset(instance_records, image_transform=image_transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_collate_batch,
        ),
    }

    if class_dir and class_prompt:
        class_records = build_media_records(
            class_dir,
            prompt=class_prompt,
            kind="class",
            media_mode=media_mode,
            recursive=recursive,
        )
        payload["class_records"] = class_records
        payload["class_loader"] = DataLoader(
            PersonalizationDataset(class_records, image_transform=image_transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_collate_batch,
        )
    else:
        payload["class_records"] = []
        payload["class_loader"] = None

    return payload


def write_manifest(path: str, records: list[ConceptMediaRecord]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import json

    with open(path, "w", encoding="utf-8") as handle:
        json.dump([record.to_json() for record in records], handle, indent=2)
