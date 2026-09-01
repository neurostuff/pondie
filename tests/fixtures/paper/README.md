# One paper, and the record built against it

`xevP8UDRAVh9` — *Reduced Intrinsic Connectivity of Amygdala in Adults with Major Depressive
Disorder*, [10.3389/fpsyt.2013.00135](https://doi.org/10.3389/fpsyt.2013.00135), PMC3801154.
**CC-BY 3.0**, which is why the text can be here at all.

`text.tables.txt` is the local build: pubget's own stylesheet over the article XML, with the
coordinate tables inlined at the position they appear. It is *the* document every
`start_char` in the record addresses, and `extraction_metadata.source_text_hash` is its
sha256 — the record is `benchmarks/gold/xevP8UDRAVh9.extraction.json`, which already ships.

## Why the text is a fixture and not a corpus path

A record and the text it was extracted from are one artefact. Split across a checkout, they
drift: the examples under `tests/fixtures/examples/` were built by `review-bootstrap-0.1.0`
against a `text.tables.txt` that no longer exists anywhere — a later pubget commit inlines
tables differently — so seventeen tests over spans, offsets and the section index could not
run in any checkout, and had not for a long time.

Shipping the pair fixes that for good. The tests that use it assert things no other test
can: that every span addresses the text it claims to, that the recorded hash still matches,
that the section index covers every span, and that the validator notices when a span is
shifted by one character.
