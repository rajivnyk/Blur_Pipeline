# import json
# import tempfile
# import unittest
# from pathlib import Path

# import numpy as np
# from PIL import Image

# from app.old.generate_data import generate_dataset
# from app.old.modeling import OldPhotoDamageModel, predict_image
# from app.old.pipeline import OldPhotoRestorationPipeline


# class OldPhotoPipelineTests(unittest.TestCase):
#     def test_dataset_generation_writes_manifest_and_masks(self) -> None:
#         with tempfile.TemporaryDirectory() as temp_dir:
#             root = Path(temp_dir)
#             input_dir = root / "input"
#             output_dir = root / "generated"
#             input_dir.mkdir(parents=True, exist_ok=True)

#             for index in range(3):
#                 image = Image.fromarray(np.full((64, 64, 3), 60 + index * 40, dtype=np.uint8))
#                 image.save(input_dir / f"{index}.png")

#             generate_dataset(input_dir=input_dir, output_dir=output_dir, n_samples=6, image_size=64, seed=7)

#             manifest_path = output_dir / "metadata" / "manifest.json"
#             self.assertTrue(manifest_path.exists())

#             manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
#             self.assertEqual(len(manifest["samples"]), 6)
#             self.assertTrue((output_dir / "masks" / "000000.png").exists())

#     def test_predict_image_returns_router_fields(self) -> None:
#         model = OldPhotoDamageModel(pretrained=False)
#         image = Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8))
#         result = predict_image(model, image, device="cpu", threshold=0.5)

#         self.assertIn("predicted_types", result.__dict__)
#         self.assertEqual(result.damage_mask.shape, (256, 256))
#         self.assertTrue(result.recommended_steps or result.predicted_types)

#     def test_pipeline_creates_artifacts(self) -> None:
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pipeline = OldPhotoRestorationPipeline(
#                 model_path=Path(temp_dir) / "missing_model.pth",
#                 runs_dir=Path(temp_dir) / "runs",
#                 device="cpu",
#             )
#             image = Image.fromarray(np.full((256, 256, 3), 180, dtype=np.uint8))
#             artifacts = pipeline.run(image)

#             self.assertTrue(artifacts.final_image_path.exists())
#             self.assertTrue(artifacts.metadata_path.exists())
#             self.assertIn("predictions", json.loads(artifacts.metadata_path.read_text(encoding="utf-8")))


# if __name__ == "__main__":
#     unittest.main()
