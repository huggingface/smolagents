import os
from .tools import Tool

class QRCodeTool(Tool):
    name = "QRCodeTool"
    description = "Generate a QR code image from any text or URL. Useful for sharing links to mobile devices."
    inputs = {
        "data": {
            "type": "string",
            "description": "The text or URL to encode (e.g., 'https://arxiv.org/pdf/1706.03762')."
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        """
        Initialize the QR Code Generator tool.
        """
        super().__init__(**kwargs)

    def forward(self, data: str) -> str:
        """
        Generates a QR Code for the given text or URL and saves it as an image.
        Useful for sharing links to mobile devices.
        
        Args:
            data: The text or URL to encode (e.g., 'https://arxiv.org/pdf/1706.03762').
        """
        try:
            # Lazy Import
            import qrcode
        except ImportError:
            return "Error: Please install 'qrcode[pil]' to use this tool."

        try:
            # 1. Create QR Code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(data)
            qr.make(fit=True)

            # 2. Generate Image
            img = qr.make_image(fill_color="black", back_color="white")
            
            # 3. Save to temporary path (or current directory)
            filename = "qrcode_output.png"
            img.save(filename)
            
            # 4. Return the path (The Agent UI will often render local image paths)
            return f"✅ QR Code generated successfully! Saved to: {os.path.abspath(filename)}"

        except Exception as e:
            return f"Error generating QR code: {str(e)}"
