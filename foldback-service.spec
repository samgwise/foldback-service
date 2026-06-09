# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for building the Foldback Service executable.

Usage:
    uv run pyinstaller foldback-service.spec

This will produce dist\foldback-service\foldback-service.exe which can be
managed by the student management app as a standalone process.

Note: The .env file must be placed in the same directory as the executable
or the working directory when running the exe.
"""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules

a = Analysis(
    ['src\\main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('.env.example', '.'),
    ],
    hiddenimports=[
        # FastAPI and ASGI
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
        'fastapi.applications',
        'starlette',
        'python_multipart',
        # Pydantic
        'pydantic',
        'pydantic_core',
        # HTTP
        'httpx',
        'httpcore',
        'h11',
        # Ollama
        'ollama',
        # Dotenv
        'dotenv',
        'python_dotenv',
        # Hugging Face / SpeechBrain
        'huggingface_hub',
        'speechbrain',
        'torchaudio',
        'torch',
        'whisperx',
        'imageio_ffmpeg',
        # Our own modules
        'src',
        'src.config',
        'src.models',
        'src.providers',
        'src.pipeline',
        'src.task_manager',
        'src.stt',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'scipy',
        'sklearn',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='foldback-service',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='foldback-service',
)
