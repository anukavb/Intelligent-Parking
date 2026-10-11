from src.zones import load_zones, save_zones, split_row


def test_split_row_interpolates_both_edges_in_point_order():
    row = [[0, 0], [90, 10], [80, 50], [-10, 40]]
    zones = split_row(row, 3)
    assert zones == [
        {"points": [[0, 0], [30, 3], [20, 43], [-10, 40]]},
        {"points": [[30, 3], [60, 7], [50, 47], [20, 43]]},
        {"points": [[60, 7], [90, 10], [80, 50], [50, 47]]},
    ]


def test_split_row_rejects_invalid_arguments():
    try:
        split_row([[0, 0]] * 4, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero slots must be rejected")


def test_zone_json_round_trip(tmp_path):
    path = tmp_path / "nested" / "zones.json"
    zones = split_row([[0, 0], [30, 0], [30, 10], [0, 10]], 2)
    save_zones(path, zones)
    assert load_zones(path) == zones
