import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Prétraitement dataset images pour VisionAI")
    parser.add_argument("input_dir", type=str, help="Dossier racine du dataset (1 sous-dossier = 1 classe)")
    parser.add_argument("--size", type=int, default=64, help="Taille cible (carré) en pixels (défaut: 64)")
    parser.add_argument("--mode", type=str, choices=["grayscale", "rgb"], default="rgb", help="Mode couleur (défaut: rgb)")
    parser.add_argument("--split", type=float, default=0.2, help="Ratio test (défaut: 0.2 = 80/20)")
    parser.add_argument("--output_dir", type=str, default=None, help="Dossier de sortie (défaut: même dossier que input_dir)")
    return parser.parse_args()


def load_class_paths(input_dir: Path) -> dict[str, list[Path]]:
    class_paths = {}
    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue
        images = [
            p for p in subdir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        ]
        if images:
            class_paths[subdir.name] = sorted(images)
    return class_paths


def process_image(path: Path, size: int, mode: str) -> np.ndarray:
    img = Image.open(path).convert("L" if mode == "grayscale" else "RGB")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.flatten()


def build_dataset(class_paths: dict[str, list[Path]], size: int, mode: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    class_names = sorted(class_paths.keys())
    label_map = {name: idx for idx, name in enumerate(class_names)}

    vectors = []
    labels = []
    errors = 0

    for class_name, paths in class_paths.items():
        for path in paths:
            try:
                vec = process_image(path, size, mode)
                vectors.append(vec)
                labels.append(label_map[class_name])
            except Exception:
                errors += 1

    if errors:
        print(f"⚠️  {errors} image(s) ignorée(s) (format non supporté ou corrompue)")

    return np.array(vectors, dtype=np.float32), np.array(labels, dtype=np.int32), class_names


def split_dataset(X: np.ndarray, y: np.ndarray, test_ratio: float) -> tuple:
    min_class_count = int(np.min(np.bincount(y))) if len(y) > 0 else 0
    stratify = y if len(np.unique(y)) > 1 and min_class_count >= 2 else None
    return train_test_split(X, y, test_size=test_ratio, random_state=42, stratify=stratify)


def save_outputs(X_train, X_test, y_train, y_test, class_names: list[str], size: int, mode: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_count = X_train.shape[1]
    cols = [f"px_{i}" for i in range(feature_count)]

    for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        df = pd.DataFrame(X, columns=cols)
        df.insert(0, "label", y)
        df.to_csv(output_dir / f"{split}.csv", index=False)

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)

    meta = {
        "classes": class_names,
        "image_size": size,
        "mode": mode,
        "feature_dim": feature_count,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    pd.Series(meta).to_json(output_dir / "meta.json")


def print_summary(X_train, X_test, y_train, y_test, class_names: list[str], size: int, mode: str, output_dir: Path):
    channels = 1 if mode == "grayscale" else 3
    total = len(y_train) + len(y_test)

    print("\n" + "=" * 50)
    print("  RÉSUMÉ DU PRÉTRAITEMENT — VisionAI")
    print("=" * 50)
    print(f"  Images traitées       : {total}")
    print(f"  Taille cible          : {size}x{size} px")
    print(f"  Mode couleur          : {mode} ({channels} canal(aux))")
    print(f"  Dimensions vecteur    : {X_train.shape[1]}")
    print(f"  Train / Test          : {len(y_train)} / {len(y_test)}")
    print(f"  Classes ({len(class_names)})          : {', '.join(class_names)}")
    print(f"  Sortie                : {output_dir}")
    print("=" * 50)

    print("\n  Répartition par split :")
    for split_name, y in [("Train", y_train), ("Test", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        detail = "  |  ".join(f"{class_names[c]}: {n}" for c, n in zip(unique, counts))
        print(f"    {split_name}: {detail}")
    print()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()

    if not input_dir.exists():
        print(f"❌ Dossier introuvable : {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir.parent / "processed"

    print(f"📂 Chargement depuis : {input_dir}")
    class_paths = load_class_paths(input_dir)

    if not class_paths:
        print("❌ Aucune classe trouvée. Vérifiez que le dossier contient des sous-dossiers avec des images.")
        sys.exit(1)

    total_images = sum(len(v) for v in class_paths.values())
    print(f"🔍 {len(class_paths)} classe(s) détectée(s) — {total_images} image(s) au total")

    print(f"⚙️  Traitement en cours ({args.mode}, {args.size}x{args.size})...")
    X, y, class_names = build_dataset(class_paths, args.size, args.mode)

    if len(X) == 0:
        print("❌ Aucune image n'a pu être traitée. Vérifiez que les fichiers ne sont pas corrompus.")
        sys.exit(1)

    print(f"✂️  Split train/test ({int((1-args.split)*100)}/{int(args.split*100)})...")
    X_train, X_test, y_train, y_test = split_dataset(X, y, args.split)

    print(f"💾 Sauvegarde vers : {output_dir}")
    save_outputs(X_train, X_test, y_train, y_test, class_names, args.size, args.mode, output_dir)

    print_summary(X_train, X_test, y_train, y_test, class_names, args.size, args.mode, output_dir)


if __name__ == "__main__":
    main()
