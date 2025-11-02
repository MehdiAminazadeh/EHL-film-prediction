import base64
import io
from pathlib import Path
from PIL import Image
import streamlit as st


def show_glowing_logo(image_path: str = "assets/megt_logo.jpg"):
    """
    Displays the MEGT logo with transparent background and glowing red effect.
    """

    def remove_white_background(path: Path) -> bytes:
        img = Image.open(path).convert("RGBA")
        pixels = img.getdata()
        new_pixels = []
        for r, g, b, a in pixels:
            if r > 240 and g > 240 and b > 240:
                new_pixels.append((255, 255, 255, 0))
            else:
                new_pixels.append((r, g, b, a))
        img.putdata(new_pixels)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    img_path = Path(image_path)
    if not img_path.exists():
        st.warning(f"Logo not found at {img_path.resolve()}")
        return

    logo_bytes = remove_white_background(img_path)
    logo_b64 = base64.b64encode(logo_bytes).decode("utf-8")

    st.markdown(
        """
        <style>
        .megt-logo-wrap {
            display: flex;
            justify-content: center;
            margin-bottom: 1.5rem;
        }
        .megt-logo {
            width: 480px;              /* Increased width */
            height: auto;              /* Keep aspect ratio */
            max-width: 90vw;           /* Responsive */
            transform: scaleX(1.25);   /* Stretches the logo horizontally */
            filter:
                drop-shadow(0 0 10px rgba(198, 55, 44, 1))
                drop-shadow(0 0 25px rgba(198, 55, 44, 0.8))
                drop-shadow(0 0 45px rgba(198, 55, 44, 0.6));
        }
        </style>
        """,
    unsafe_allow_html=True,
)


    st.markdown(
        f"""
        <div class="megt-logo-wrap">
            <img src="data:image/png;base64,{logo_b64}" class="megt-logo" />
        </div>
        """,
        unsafe_allow_html=True,
    )
