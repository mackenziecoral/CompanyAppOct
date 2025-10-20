import importlib.machinery
import json
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_stub(name: str, module: types.ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = module


class _DummyGeoDataFrame:
    def __init__(self, *args, **kwargs):
        self.columns = []

    def to_crs(self, *args, **kwargs):
        return self

    def to_json(self):
        return "{}"

    @property
    def empty(self):
        return True

    def total_bounds(self):
        return (0, 0, 0, 0)


geopandas_stub = types.ModuleType("geopandas")
geopandas_stub.GeoDataFrame = _DummyGeoDataFrame
geopandas_stub.read_file = lambda *args, **kwargs: _DummyGeoDataFrame()
_ensure_stub("geopandas", geopandas_stub)


class _PolarsSeries:
    def __init__(self, values, name: str | None = None):
        if isinstance(values, _PolarsSeries):
            self._values = list(values._values)
            self.name = name or values.name
        else:
            self._values = list(values)
            self.name = name

    def cast(self, *_args, **_kwargs):
        return _PolarsSeries(self._values, name=self.name)

    @property
    def str(self):
        return _PolarsStringMethods(self)

    def alias(self, name: str):
        return _PolarsSeries(self._values, name=name)


class _PolarsStringMethods:
    def __init__(self, series: _PolarsSeries):
        self._series = series

    def replace_all(self, pattern: str, replacement: str):
        regex = re.compile(pattern)
        updated = [regex.sub(replacement, str(value)) if value is not None else None for value in self._series._values]
        return _PolarsSeries(updated, name=self._series.name)

    def to_uppercase(self):
        updated = [value.upper() if isinstance(value, str) else value for value in self._series._values]
        return _PolarsSeries(updated, name=self._series.name)


class _PolarsDataFrame:
    def __init__(self, rows=None, schema=None):
        rows = rows or []
        schema = list(schema or [])
        self._columns = schema
        self._rows: list[tuple] = []
        for row in rows:
            if isinstance(row, dict):
                self._rows.append(tuple(row.get(col) for col in self._columns))
            else:
                self._rows.append(tuple(row))

    @property
    def columns(self):
        return list(self._columns)

    def __getitem__(self, item):
        idx = self._columns.index(item)
        return _PolarsSeries([row[idx] for row in self._rows], name=item)

    def with_columns(self, updates):
        table = {name: [] for name in self._columns}
        for row in self._rows:
            for idx, col in enumerate(self._columns):
                table[col].append(row[idx])
        for series in updates:
            name = getattr(series, "name", None)
            if name is None:
                continue
            table[name] = list(series._values)
            if name not in self._columns:
                self._columns.append(name)
        length = len(next(iter(table.values()), []))
        self._rows = [tuple(table[col][i] for col in self._columns) for i in range(length)]
        return self

    def is_empty(self):
        return not self._rows

    @property
    def height(self):
        return len(self._rows)

    def write_parquet(self, path):
        payload = {"columns": self._columns, "rows": self._rows}
        Path(path).write_text(json.dumps(payload))


polars_stub = types.ModuleType("polars")
polars_stub.DataFrame = _PolarsDataFrame
polars_stub.Series = _PolarsSeries


def _stub_read_parquet(path):
    payload = json.loads(Path(path).read_text())
    return _PolarsDataFrame(payload.get("rows", []), schema=payload.get("columns", []))


polars_stub.read_parquet = _stub_read_parquet
polars_stub.Utf8 = str
polars_stub.__spec__ = importlib.machinery.ModuleSpec("polars", loader=None)
_ensure_stub("polars", polars_stub)

pandas_stub = types.ModuleType("pandas")


class _DummyPandasFrame:
    def __init__(self, *args, **kwargs):
        self.columns = []
        self.empty = True

    def copy(self):
        return self


pandas_stub.DataFrame = _DummyPandasFrame
pandas_stub.Series = lambda *args, **kwargs: []
pandas_stub.read_excel = lambda *args, **kwargs: _DummyPandasFrame()
_ensure_stub("pandas", pandas_stub)

pydeck_stub = types.ModuleType("pydeck")
pydeck_stub.Layer = lambda *args, **kwargs: {}
pydeck_stub.Deck = lambda *args, **kwargs: types.SimpleNamespace()
pydeck_stub.ViewState = lambda *args, **kwargs: types.SimpleNamespace()
_ensure_stub("pydeck", pydeck_stub)


class _StreamlitExpander:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _identity_decorator(*_args, **_kwargs):
    def wrapper(func):
        return func

    return wrapper


streamlit_stub = types.ModuleType("streamlit")
streamlit_stub.cache_data = _identity_decorator
streamlit_stub.secrets = {}
streamlit_stub.info = lambda *args, **kwargs: None
streamlit_stub.warning = lambda *args, **kwargs: None
streamlit_stub.error = lambda *args, **kwargs: None
streamlit_stub.success = lambda *args, **kwargs: None
streamlit_stub.set_page_config = lambda *args, **kwargs: None
streamlit_stub.title = lambda *args, **kwargs: None
streamlit_stub.write = lambda *args, **kwargs: None
streamlit_stub.markdown = lambda *args, **kwargs: None
streamlit_stub.expander = lambda *args, **kwargs: _StreamlitExpander()
streamlit_stub.stop = lambda: None
streamlit_stub.multiselect = lambda *args, **kwargs: []
streamlit_stub.date_input = lambda *args, **kwargs: (None, None)
streamlit_stub.caption = lambda *args, **kwargs: None
streamlit_stub.columns = lambda *args, **kwargs: [types.SimpleNamespace(), types.SimpleNamespace()]
streamlit_stub.header = lambda *args, **kwargs: None
streamlit_stub.dataframe = lambda *args, **kwargs: None
streamlit_stub.download_button = lambda *args, **kwargs: None
streamlit_stub.tabs = lambda *args, **kwargs: []
streamlit_stub.sidebar = types.SimpleNamespace()
_ensure_stub("streamlit", streamlit_stub)

shapely_module = types.ModuleType("shapely")
geometry_module = types.ModuleType("shapely.geometry")
geometry_module.Point = lambda coords: coords
shapely_module.geometry = geometry_module
_ensure_stub("shapely", shapely_module)
_ensure_stub("shapely.geometry", geometry_module)

plotly_module = types.ModuleType("plotly")
graph_objects_module = types.ModuleType("plotly.graph_objects")


class _DummyFigure:
    def add_trace(self, *args, **kwargs):
        pass

    def update_layout(self, *args, **kwargs):
        pass

    def update_yaxes(self, *args, **kwargs):
        pass


graph_objects_module.Figure = _DummyFigure
plotly_module.graph_objects = graph_objects_module
_ensure_stub("plotly", plotly_module)
_ensure_stub("plotly.graph_objects", graph_objects_module)

import app


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.description = [
            ("UWI",),
            ("GSL_UWI",),
            ("SURFACE_LATITUDE",),
            ("SURFACE_LONGITUDE",),
            ("BOTTOM_HOLE_LATITUDE",),
            ("BOTTOM_HOLE_LONGITUDE",),
            ("GSL_FULL_LATERAL_LENGTH",),
            ("ABANDONMENT_DATE",),
            ("WELL_NAME",),
            ("CURRENT_STATUS",),
            ("OPERATOR_CODE",),
            ("CONFIDENTIAL_TYPE",),
            ("STRAT_UNIT_ID",),
            ("SPUD_DATE",),
            ("FIRST_PROD_DATE",),
            ("FINAL_TD",),
            ("PROVINCE_STATE",),
            ("COUNTRY",),
            ("FIELD_NAME",),
        ]

    def execute(self, query):
        self.last_query = query

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class FakeConnection:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return FakeCursor(self._rows)

    def close(self):
        pass


class FakeOracle:
    AUTH_MODE_DEFAULT = 0

    def __init__(self, rows):
        self._rows = rows
        self.defaults = types.SimpleNamespace(config_dir=None, wallet_location=None)

    def connect(self, **_kwargs):
        return FakeConnection(self._rows)


def test_try_build_parquet_from_oracle(tmp_path, monkeypatch):
    rows = [
        (
            "100011234500W400",
            "100011234500W400",
            45.0,
            -110.0,
            44.5,
            -109.5,
            5000.0,
            None,
            "Well A",
            "Producing",
            "OP",
            None,
            "STRAT",
            None,
            None,
            12000.0,
            "TX",
            "USA",
            "Field A",
        )
    ]

    cache_path = tmp_path / "processed_app_data.parquet"
    monkeypatch.setattr(app, "PROCESSED_PARQUET", cache_path, raising=False)
    monkeypatch.setattr(app, "oracledb", FakeOracle(rows), raising=False)
    monkeypatch.setattr(app, "ORACLE_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(app, "USING_CX_ORACLE_SHIM", False, raising=False)
    monkeypatch.setattr(app, "_configure_oracle_client", lambda: None)
    monkeypatch.setattr(
        app,
        "_oracle_connection_details",
        lambda: {"user": "u", "password": "p", "dsn": "host:1521/service", "mode": None},
    )
    monkeypatch.setattr(app, "_oracle_runtime_options", lambda: {})

    success, message = app._try_build_parquet_from_oracle(force=True)

    assert success, message
    assert cache_path.exists()
    frame = app.pl.read_parquet(cache_path)
    assert frame.height == 1
