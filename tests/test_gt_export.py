"""Check the two frame conventions required by the MOT and MDA readers."""
from build_mot_gt_from_xml import write_sequence


def test_gt_export_offsets_only_frames_and_limits_smoke_test(tmp_path):
    xml = tmp_path / "sample.xml"
    xml.write_text('<annotations><track id="4" label="car">'
                   '<box frame="0" outside="0" xtl="10" ytl="20" xbr="30" ybr="50"/>'
                   '<box frame="1" outside="0" xtl="11" ytl="20" xbr="31" ybr="50"/>'
                   '</track></annotations>', encoding="utf-8")
    mot, mda = tmp_path / "mot.txt", tmp_path / "mda.txt"
    write_sequence(xml, mot, frame_offset=0, max_frames=1)
    write_sequence(xml, mda, frame_offset=1, max_frames=1)
    assert mot.read_text().splitlines() == ["0,5,10,20,20,30,1,1,1"]
    assert mda.read_text().splitlines() == ["1,5,10,20,20,30,1,1,1"]
