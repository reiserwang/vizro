# -*- mode: python ; coding: utf-8 -*-
# Vizro Analytics — PyInstaller spec
# Target: macOS 12+ (Tahoe), end-user distribution
# Strategy: exclude torch/pyarrow/gradio/dev-tools; enable strip+compression
from PyInstaller.utils.hooks import collect_data_files

datas = [('src/ui/frontend', 'src/ui/frontend')]

# ── Complete exclusion list ──────────────────────────────────────────────────
# Heavy optional deps: torch (276 MB) excluded because LSTM is not in native app
# pyarrow (114 MB) excluded because Parquet import is optional at runtime
# Dev-only packages that should NEVER be bundled
# Gradio dragged in transitively by some imports — excluded since native uses FastAPI/HTML
EXCLUDES = [
    # Deep learning — LSTM disabled in native app (276 MB savings)
    'torch', 'torchvision', 'torchaudio', 'torch.nn', 'torch.optim',
    'torch.utils', 'torch.cuda', 'torch.distributed',

    # Parquet/Arrow — optional data format (114 MB savings)
    'pyarrow', 'pyarrow.parquet', 'pyarrow.compute',

    # Gradio — web-only UI; native app uses FastAPI + HTML frontend
    'gradio', 'gradio_client', 'safehttpx', 'python-multipart',
    'ffmpy', 'pydub', 'aiofiles', 'uvloop',

    # Image processing (pulled in transitively via gradio/plotly)
    'PIL', 'Pillow', 'cv2', 'skimage',

    # Dev tools — MUST NOT be bundled in production
    'jedi', 'jedi.api', 'jedi.plugins',
    'mypy', 'mypy_extensions',
    'black', 'blib2to3', 'pathspec',
    'flake8', 'pyflakes', 'pycodestyle', 'mccabe',
    'pytest', '_pytest', 'py', 'pluggy',
    'isort', 'autopep8',

    # Jupyter / IPython ecosystem (sometimes pulled in)
    'IPython', 'ipykernel', 'ipython_genutils', 'jupyter',
    'jupyterlab', 'notebook', 'nbformat', 'nbconvert',
    'traitlets', 'tornado', 'zmq', 'ipywidgets',

    # Test/benchmark utilities
    'hypothesis', 'coverage', 'codecov',

    # Unused optional Vizro / Dash / werkzeug
    'vizro', 'dash', 'werkzeug', 'flask', 'flask_compress',

    # Cloud SDKs (not needed locally)
    'boto3', 'botocore', 's3transfer', 'awscli',
    'google.cloud', 'azure',

    # Unused matplotlib backends (keep core, drop GUI toolkits)
    'matplotlib.backends.backend_gtk3agg',
    'matplotlib.backends.backend_gtk3cairo',
    'matplotlib.backends.backend_wxagg',
    'matplotlib.backends.backend_tkagg',

    # Unused stdlib modules that PyInstaller sometimes includes
    'tkinter', '_tkinter', 'tk', 'tcl',
    'xmlrpc', 'imaplib', 'smtplib', 'ftplib',
    'curses', 'readline',
]

a = Analysis(
    ['main.py'],
    pathex=['src'],
    binaries=[],
    datas=datas,
    hiddenimports=[
        # These are dynamically imported at runtime — must be listed explicitly
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi',
        'starlette',
        'starlette.routing',
        'starlette.responses',
        'starlette.staticfiles',
        'webview',
        'webview.menu',
        'webview.platforms.cocoa',
        'statsmodels.tsa.arima.model',
        'statsmodels.tsa.statespace.sarimax',
        'statsmodels.tsa.vector_ar.var_model',
        'statsmodels.tsa.statespace.structural',
        'statsmodels.tsa.stattools',
        'sklearn.linear_model',
        'sklearn.metrics',
        'sklearn.decomposition',
        'scipy.stats',
        'causalnex.structure.notears',
        'causalnex.network',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=1,          # Level 1: strips asserts but preserves docstrings (numpy/scipy require them)
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Vizro Analytics',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,           # Strip debug symbols from binaries
    upx=True,             # UPX compress all binaries
    upx_exclude=[
        # Skip UPX on libs that break when compressed
        'libpython*.dylib',
        'Python',
    ],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,     # Universal binary (arm64 + x86_64) — let PyInstaller detect
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=['libpython*.dylib', 'Python'],
    name='Vizro Analytics',
)

app = BUNDLE(
    coll,
    name='Vizro Analytics.app',
    icon=None,
    bundle_identifier='com.vizro.analytics',
    version='1.0.0',
    info_plist={
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '12.0',   # macOS Monterey+
        'NSRequiresAquaSystemAppearance': False,  # Support dark mode
        'CFBundleShortVersionString': '1.0.0',
    },
)
