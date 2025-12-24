import sys
import unittest
import os
from unittest.mock import MagicMock, patch
from smolagents.qrcode_tool import QRCodeTool

class TestQRCodeTool(unittest.TestCase):
    def test_qr_generation(self):
        """
        Test QR code generation logic with mocked library.
        """
        # 1. Mock the Image Object
        mock_img = MagicMock()
        mock_img.save = MagicMock()

        # 2. Mock the QRCode Object
        mock_qr_instance = MagicMock()
        mock_qr_instance.make_image.return_value = mock_img

        # 3. Mock the Library
        mock_qrcode = MagicMock()
        mock_qrcode.QRCode.return_value = mock_qr_instance
        mock_qrcode.constants.ERROR_CORRECT_L = 1

        # 4. Patch and Run
        with patch.dict(sys.modules, {'qrcode': mock_qrcode}):
            tool = QRCodeTool()
            result = tool.forward("https://srijan.ai")
            
            # 5. Verify
            self.assertIn("QR Code generated successfully", result)
            self.assertIn("qrcode_output.png", result)
            mock_qr_instance.add_data.assert_called_with("https://srijan.ai")
            mock_img.save.assert_called_once()

    def test_missing_dependency(self):
        """Test graceful failure."""
        with patch.dict(sys.modules):
            if 'qrcode' in sys.modules:
                del sys.modules['qrcode']
            
            tool = QRCodeTool()
            result = tool.forward("test")
            self.assertIn("Please install 'qrcode[pil]'", result)

if __name__ == "__main__":
    unittest.main()
