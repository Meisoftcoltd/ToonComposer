import sys
import unittest
from unittest.mock import MagicMock, patch

# 1. Mock heavyweight dependencies before importing app.py
# This prevents errors from missing packages like torch, cv2, etc.
class MockImage:
    class Image:
        def convert(self, mode): return self
        def resize(self, size): return self
    @staticmethod
    def new(*args, **kwargs):
        m = MagicMock(spec=MockImage.Image)
        m.convert.return_value = m
        m.resize.return_value = m
        return m
    @staticmethod
    def fromarray(*args, **kwargs):
        m = MagicMock(spec=MockImage.Image)
        m.convert.return_value = m
        m.resize.return_value = m
        return m
    @staticmethod
    def alpha_composite(*args, **kwargs):
        m = MagicMock(spec=MockImage.Image)
        m.convert.return_value = m
        m.resize.return_value = m
        return m

# Mock numpy with a class that supports some operations to reduce brittleness
class MockNumpy:
    def array(self, *args, **kwargs):
        m = MagicMock()
        m.shape = (480, 832, 4)
        m.__getitem__.return_value = m
        m.__lt__.return_value = m
        m.__gt__.return_value = m
        m.__and__.return_value = m
        return m
    def all(self, *args, **kwargs): return True
    def zeros(self, *args, **kwargs): return MagicMock()
    @property
    def uint8(self): return 'uint8'

sys.modules['numpy'] = MockNumpy()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MockImage
sys.modules['PIL'].Image = MockImage
sys.modules['torch'] = MagicMock()
sys.modules['cv2'] = MagicMock()
mock_gradio = MagicMock()
mock_gradio.__version__ = "5.25.2"
sys.modules['gradio'] = mock_gradio
sys.modules['einops'] = MagicMock()
sys.modules['huggingface_hub'] = MagicMock()
sys.modules['util.stylesheets'] = MagicMock()
sys.modules['tooncomposer'] = MagicMock()

mock_argparse = MagicMock()
sys.modules['argparse'] = mock_argparse
mock_cli_args = MagicMock()
mock_cli_args.dtype = 'float32'
mock_cli_args.resolution = '480p'
mock_cli_args.device = 'cpu'
mock_cli_args.wan_model_dir = None
mock_cli_args.tooncomposer_dir = None
mock_cli_args.hf_token = None
mock_cli_args.fast_dev = True
mock_argparse.ArgumentParser.return_value.parse_args.return_value = mock_cli_args

# Mock open to prevent FileNotFoundError during app.py initialization
import builtins
real_open = builtins.open
def mocked_open(path, *args, **kwargs):
    if "config.json" in str(path):
        m = MagicMock()
        m.__enter__.return_value.read.return_value = "{}"
        return m
    return real_open(path, *args, **kwargs)

# Perform the import with necessary patches
with patch('builtins.open', side_effect=mocked_open), \
     patch('app.initialize_model', return_value=(MagicMock(), 'cpu', 'float32')), \
     patch('app.resolve_tooncomposer_repo_dir', return_value='/tmp'), \
     patch('app.build_checkpoints_by_resolution', return_value={}):
    import app

class TestAppUtils(unittest.TestCase):

    def test_create_blank_mask(self):
        with patch('app.Image.new') as mock_new:
            app.create_blank_mask(100, 50)
            mock_new.assert_called_with('RGB', (100, 50), color='white')

    def test_create_mask_with_sketch_none(self):
        with patch('app.create_blank_mask') as mock_blank:
            mock_blank.return_value = "blank_mask"
            result = app.create_mask_with_sketch(None, 100, 50)
            mock_blank.assert_called_with(100, 50)
            self.assertEqual(result, "blank_mask")

    def test_create_mask_with_sketch_empty_layers(self):
        sketch = {'layers': []}
        with patch('app.create_blank_mask') as mock_blank:
            mock_blank.return_value = "blank_mask"
            result = app.create_mask_with_sketch(sketch, 100, 50)
            mock_blank.assert_called_with(100, 50)
            self.assertEqual(result, "blank_mask")

    def test_create_mask_with_sketch_dict_processing(self):
        mock_layer = MagicMock(spec=MockImage.Image)
        mock_layer.convert.return_value = mock_layer
        sketch = {'layers': [mock_layer]}

        with patch('numpy.zeros') as mock_zeros, \
             patch('app.Image.fromarray') as mock_from_array:

            mock_final_img = MagicMock(spec=MockImage.Image)
            mock_final_img.resize.return_value = "processed_mask"
            mock_from_array.return_value = mock_final_img

            result = app.create_mask_with_sketch(sketch, 832, 480)

            self.assertEqual(result, "processed_mask")
            mock_from_array.assert_called()
            mock_final_img.resize.assert_called_with((832, 480))

    def test_create_mask_with_sketch_pil_image_ghosting(self):
        mock_sketch = MagicMock(spec=MockImage.Image)
        mock_sketch.mode = 'RGB'
        mock_sketch.resize.return_value = mock_sketch
        mock_sketch.convert.return_value = mock_sketch

        with patch('app.Image.new') as mock_img_new, \
             patch('app.Image.alpha_composite') as mock_composite:

            mock_overlay = MagicMock(spec=MockImage.Image)
            mock_img_new.return_value = mock_overlay

            mock_result_img = MagicMock(spec=MockImage.Image)
            mock_result_img.convert.return_value = "ghosted_mask"
            mock_composite.return_value = mock_result_img

            result = app.create_mask_with_sketch(mock_sketch, 100, 50)

            self.assertEqual(result, "ghosted_mask")
            mock_sketch.resize.assert_called_with((100, 50))
            mock_img_new.assert_called_with('RGBA', (100, 50), (255, 255, 255, 128))
            mock_composite.assert_called_with(mock_sketch, mock_overlay)

if __name__ == '__main__':
    unittest.main()
