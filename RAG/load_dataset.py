import os
import argparse
from embedder import embed_image
from db_conn import get_collection

collection = get_collection()

DATASET_DEFAULTS = {

    "coco": "C:\\NeuroVisionCombined4\\data\\coco_minitrain_10k\\images\\train2017",

    "ffaq": "C:\\NeuroVisionCombined4\\data\\FFAQ",

    "gopro": "C:\\NeuroVisionCombined4\\data\\gopro_deblur\\sharp\\images",
    
    "lol_real": "C:\\NeuroVisionCombined4\\data\\LOL-v2\\Real_captured\\Train\\Normal",

    "lol_synthetic": "C:\\NeuroVisionCombined4\\data\\LOL-v2\\Synthetic\\Train\\Normal",

    "reside": "C:\\NeuroVisionCombined4\\data\\RESIDE-6K\\train\\GT",

    "reside_its": "C:\\NeuroVisionCombined4\\data\\RESIDE-ITS\\clear"
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def add_images_from_folder(folder_path: str, source_name: str):

    if not os.path.isdir(folder_path):
        print(f"[load_dataset] Folder not found: {folder_path}")
        return 0

    count = 0

    # Set max limit for specific datasets
    max_images = None
    if source_name in {"coco", "ffaq"}:
        max_images = 3000

    for file in sorted(os.listdir(folder_path)):

        if os.path.splitext(file)[1].lower() not in IMAGE_EXTS:
            continue

        # Stop if max_images limit is reached
        if max_images is not None and count >= max_images:
            break

        path = os.path.join(folder_path, file)

        try:
            embedding = embed_image(path)

            collection.add(
                ids=[f"{source_name}_{count}"],
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "image_path": path,
                    "dataset": source_name
                }]
            )

            count += 1

            if count % 50 == 0:
                print(f"[load_dataset] {source_name}: {count} images embedded...")

        except Exception as exc:
            print(f"[load_dataset] Skipping {path}: {exc}")

    print(f"[load_dataset] Done — {count} images stored for dataset '{source_name}'.")

    return count



def load_all_defaults():

    for name, folder in DATASET_DEFAULTS.items():

        if os.path.isdir(folder):

            print(f"\n[load_dataset] Loading '{name}' from {folder}")

            add_images_from_folder(folder, name)

        else:
            print(f"[load_dataset] Skipping '{name}' (folder not found: {folder})")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Embed reference images into ChromaDB."
    )

    parser.add_argument("--dataset", default=None)
    parser.add_argument("--folder", default=None)
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.all:

        load_all_defaults()

    elif args.dataset and args.folder:

        add_images_from_folder(args.folder, args.dataset)

    elif args.dataset and args.dataset in DATASET_DEFAULTS:

        add_images_from_folder(
            DATASET_DEFAULTS[args.dataset],
            args.dataset
        )

    else:

        add_images_from_folder(
            DATASET_DEFAULTS["lol"],
            "lol"
        )