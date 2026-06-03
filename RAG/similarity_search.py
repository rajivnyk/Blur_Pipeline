from RAG.embedder import embed_image
from RAG.db_conn import get_collection
from collections import Counter
from PIL import Image

collection = get_collection()


def find_similar_images(
    query_image,
    k: int = 3,
    dataset: str = None
):

    query_embedding = embed_image(query_image)

    where_filter = None

    if dataset:
        where_filter = {
            "dataset": dataset
        }

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        where=where_filter
    )

    output = []

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for meta, dist in zip(metadatas, distances):

        output.append(
            (
                meta["image_path"],
                meta["dataset"],
                float(dist)
            )
        )

    return output
    return output


def get_image_context(img: Image.Image, k: int = 3) -> str:
    """Queries ChromaDB and returns the dominant dataset type for the uploaded image."""
    try:
        results = find_similar_images(img, k=k)
        if not results:
            return "default"
        
        # Extract the dataset name from the results (2nd element in the tuple)
        datasets = [res[1] for res in results]
        
        # Group similar styles
        style = Counter(datasets).most_common(1)[0][0]
        
        if "lol" in style:
            return "low_light"
        elif "gopro" in style:
            return "blur"
        elif "ffaq" in style:
            return "face"
        else:
            return "default"
            
    except Exception as exc:
        print(f"[RAG Error] {exc}")
        return "default"


if __name__ == "__main__":

    query_image = "C:\\NeuroVisionCombined4\\data\\gopro_deblur\\sharp\\images\\000001.png"

    results = find_similar_images(
        query_image,
        k=3
    )

    for i, (path, ds, dist) in enumerate(results, 1):

        print(
            f"{i}. {path} | {ds} | distance={dist:.4f}"
        )