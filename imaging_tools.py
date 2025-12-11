# imaging_tools.py

import io
import os
import zipfile

import streamlit as st
from PIL import Image


class ImagingToolsUI:
    """Imaging utilities for the Rootweiler 'Imaging' section."""

    @classmethod
    def render(cls):
        st.subheader("Imaging tools")

        st.markdown(
            """
            **Image / file compressor (beta)**  
            Upload images or other files, choose a target size, and download a compressed ZIP.
            
            Images are compressed by:
            - Optional resizing to a maximum dimension
            - JPEG re-encoding with adjusted quality to aim for your target size
            
            Non-image files are included unchanged in the ZIP.
            """
        )

        uploaded_files = st.file_uploader(
            "Upload one or more files",
            type=None,  # accept anything; we'll treat images specially
            accept_multiple_files=True,
        )

        if not uploaded_files:
            st.info("Add a few images or files above to get started.")
            return

        # Compression settings
        st.markdown("#### Compression settings")

        col1, col2 = st.columns(2)

        with col1:
            target_size_kb = st.slider(
                "Approximate target size per image (kB)",
                min_value=50,
                max_value=5000,
                value=500,
                step=50,
                help="The compressor will aim for this size per image by adjusting JPEG quality.",
            )

        with col2:
            resize_option = st.selectbox(
                "Max image dimension (longest side)",
                options=[
                    "No resizing",
                    "800 px",
                    "1200 px",
                    "2000 px",
                ],
                index=1,
                help="Resizing down first often gives better quality at smaller file sizes.",
            )

        max_dim = None
        if resize_option != "No resizing":
            max_dim = int(resize_option.split()[0])

        if st.button("Run compression", type="primary"):
            cls._run_compression(uploaded_files, target_size_kb, max_dim)

    # ----------------- Internals ----------------- #

    @staticmethod
    def _run_compression(uploaded_files, target_size_kb: int, max_dim: int | None):
        compressed_files: list[tuple[str, bytes]] = []
        summary_rows = []

        for f in uploaded_files:
            raw_bytes = f.getvalue()
            orig_size_kb = len(raw_bytes) / 1024.0
            ext = os.path.splitext(f.name)[1].lower()

            is_image = False
            try:
                img = Image.open(io.BytesIO(raw_bytes))
                img.verify()  # quick check
                is_image = True
            except Exception:
                is_image = False

            if is_image:
                # Re-open (verify() puts it into a closed state)
                img = Image.open(io.BytesIO(raw_bytes))
                img = img.convert("RGB")

                compressed_bytes = ImagingToolsUI._compress_image(
                    img,
                    target_size_kb=target_size_kb,
                    max_dim=max_dim,
                )

                new_name = os.path.splitext(f.name)[0] + "_compressed.jpg"
                compressed_files.append((new_name, compressed_bytes))

                new_size_kb = len(compressed_bytes) / 1024.0
                ratio = new_size_kb / orig_size_kb if orig_size_kb > 0 else 1.0

                summary_rows.append(
                    {
                        "File": new_name,
                        "Original (kB)": round(orig_size_kb, 1),
                        "Compressed (kB)": round(new_size_kb, 1),
                        "Size ratio": f"{ratio:.2f}x",
                    }
                )
            else:
                # Non-image: just include unchanged
                compressed_files.append((f.name, raw_bytes))
                summary_rows.append(
                    {
                        "File": f.name,
                        "Original (kB)": round(orig_size_kb, 1),
                        "Compressed (kB)": round(orig_size_kb, 1),
                        "Size ratio": "1.00x (unchanged)",
                    }
                )

        if not compressed_files:
            st.warning("Nothing to compress – no valid files detected.")
            return

        # Show summary table
        st.markdown("#### Compression summary")
        st.dataframe(summary_rows, use_container_width=True)

        # Build ZIP in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in compressed_files:
                zf.writestr(name, data)
        zip_buffer.seek(0)

        st.download_button(
            "Download compressed files (.zip)",
            data=zip_buffer,
            file_name="rootweiler_compressed_files.zip",
            mime="application/zip",
        )

    @staticmethod
    def _compress_image(
        img: Image.Image,
        target_size_kb: int,
        max_dim: int | None = None,
    ) -> bytes:
        """
        Resize (optional) and JPEG-compress an image, trying to hit target_size_kb.
        Returns JPEG bytes.
        """
        # Optional resize
        if max_dim is not None:
            w, h = img.size
            longest = max(w, h)
            if longest > max_dim and longest > 0:
                scale = max_dim / float(longest)
                new_size = (int(w * scale), int(h * scale))
                img = img.resize(new_size, Image.LANCZOS)

        # Binary search over JPEG quality to approach target size
        # Start with high quality, then go down as needed
        low_q, high_q = 20, 95
        best_bytes = None

        for _ in range(8):  # 8 iterations is enough for rough target
            q = (low_q + high_q) // 2

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q, optimize=True)
            data = buf.getvalue()
            size_kb = len(data) / 1024.0

            best_bytes = data  # keep last attempt

            if size_kb > target_size_kb and q > low_q:
                # file too big -> lower quality
                high_q = q - 1
            else:
                # file small enough or we can't go lower
                low_q = q + 1

        return best_bytes or b""
