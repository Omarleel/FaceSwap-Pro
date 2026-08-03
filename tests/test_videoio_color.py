from types import SimpleNamespace

from faceswap_pro.videoio import _codec_color_parameters


def test_x264_color_parameters_include_complete_vui_metadata():
    encoding = SimpleNamespace(
        color_primaries="bt709",
        color_transfer="bt709",
        color_space="bt709",
        color_range="tv",
    )

    result = _codec_color_parameters("libx264", encoding)

    assert result[0] == "-x264-params"
    assert "colorprim=bt709" in result[1]
    assert "transfer=bt709" in result[1]
    assert "colormatrix=bt709" in result[1]
    assert "range=limited" in result[1]
