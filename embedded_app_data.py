"""Embedded shapefile payload for the Streamlit app.

Populate ``EMBEDDED_APP_DATA_B64`` with a base64-encoded pickle created by
``load_and_process_all_data`` so the application can start without touching the
raw shapefiles. Leave it set to ``None`` to fall back to on-disk caching.
"""

EMBEDDED_APP_DATA_B64: str | None = None
