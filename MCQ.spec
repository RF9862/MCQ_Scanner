# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['E:/working/jaisan/barcode/Barcode-Scanner-master/mcq/MCQ.py'],
    pathex=[],
    binaries=[],
    datas=[('E:/working/jaisan/barcode/Barcode-Scanner-master/mcq/keyWindow.ui', '.'), ('E:/working/jaisan/barcode/Barcode-Scanner-master/mcq/main.ui', '.'), ('E:/working/jaisan/barcode/Barcode-Scanner-master/mcq/template', 'template/')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MCQ',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['E:\\working\\jaisan\\barcode\\Barcode-Scanner-master\\mcq\\omr.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MCQ',
)
