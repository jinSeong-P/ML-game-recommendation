"""Korean font setup for the main EDA notebook."""

def setup_korean_font() -> str | None:
    """Choose an installed Korean-capable Matplotlib font when available."""
    try:
        import matplotlib as mpl
        from matplotlib import font_manager
    except ImportError:
        return None

    for family in (
        "Malgun Gothic",
        "Noto Sans CJK KR",
        "NanumGothic",
        "Noto Sans CJK JP",
        "AppleGothic",
    ):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
        except (ValueError, RuntimeError):
            continue
        mpl.rcParams["font.family"] = family
        mpl.rcParams["axes.unicode_minus"] = False
        return family

    mpl.rcParams["axes.unicode_minus"] = False
    return None
