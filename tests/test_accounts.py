import json

from orderagent.accounts import AccountStore, initialize, inspect_users


def test_initialization_is_incremental_and_passwords_are_not_in_database(tmp_path):
    store = AccountStore(tmp_path / "accounts.sqlite3")
    path = tmp_path / "credentials.json"
    users = {"U-1": {"order_count": 1, "example_order_ids": ["O-1"]}}
    assert initialize(store, users, path)["planned_new"] == 1
    assert not path.exists()
    assert initialize(store, users, path, True)["created"] == 1
    credential = json.loads(path.read_text())[0]
    row = store.authenticate(credential["username"], credential["password"])
    assert row["user_id"] == "U-1"
    assert credential["password"] not in store.path.read_bytes().decode("latin1")
    assert store.authenticate(credential["username"], "wrong") is None
    assert store.authenticate("unknown", "wrong") is None
    store.reset(credential["username"], "replacement")
    original = path.read_bytes()
    assert initialize(store, users, path, True)["created"] == 0
    assert path.read_bytes() == original
    assert store.authenticate(credential["username"], "replacement")
    assert store.authenticate(credential["username"], credential["password"]) is None
    path.unlink()
    assert initialize(store, users, path, True)["created"] == 0
    assert not path.exists()  # Existing passwords cannot be recovered.
    with store.connect() as db:
        db.execute("UPDATE accounts SET enabled=0")
    assert store.authenticate(credential["username"], "replacement") is None


def test_data_scan_reports_orphan_and_mismatched_users(tmp_path):
    (tmp_path / "orders.json").write_text(
        json.dumps(
            [
                {"id": "O1", "user_id": "U1", "synthetic": True},
                {"id": "O2", "user_id": " bad ", "synthetic": True},
            ]
        )
    )
    (tmp_path / "tickets.json").write_text(
        json.dumps(
            [
                {"id": "T1", "order_id": "O1", "user_id": "U2"},
            ]
        )
    )
    (tmp_path / "after_sales_cases.json").write_text('[{"id":"scenario"}]')
    users, issues = inspect_users(tmp_path)
    assert list(users) == ["U1"]
    assert {x["kind"] for x in issues} == {
        "invalid_order_user",
        "orphan_user",
        "order_user_mismatch",
    }
