"""The projection keeps what makes a `local_id` resolvable."""

from pondie.schema import generate


def test_every_projected_class_keeps_local_id_as_its_identifier():
    documents, _ = generate.build_output()
    classes = {
        name: body
        for document in documents.values()
        for name, body in (document.get("classes") or {}).items()
    }
    with_local_id = {
        name: body["attributes"]["local_id"]
        for name, body in classes.items()
        if "local_id" in (body.get("attributes") or {})
    }
    assert with_local_id
    assert all(slot.get("identifier") is True for slot in with_local_id.values())
