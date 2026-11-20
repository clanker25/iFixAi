from pathlib import Path

from ifixai.rules.loader import RuleLoader


def test_load_all_rules_includes_each_shipped_family():
    root = Path(__file__).parents[1] / "inspections"
    loader = RuleLoader(root)
    plans = loader.load_all_rules()
    # One per family; B05 lost its unused steps in the cleanup, so B06 stands for the B family.
    for test_id in ("B06", "P13", "S02", "V01"):
        individually = loader.load_rules(test_id)
        assert individually.steps
        assert plans[test_id] == individually
