"""Embedded shapefile payload for the Streamlit app.

Populate ``EMBEDDED_APP_DATA_B64`` with a base64-encoded pickle created by
``load_and_process_all_data`` so the application can start without touching the
raw shapefiles. The helper automatically stamps the payload with the current
cache version, so regenerate this value any time shapefile processing changes.

Leave ``EMBEDDED_APP_DATA_B64`` set to ``None`` to fall back to on-disk caching.
"""

EMBEDDED_APP_DATA_B64: str | None = None
