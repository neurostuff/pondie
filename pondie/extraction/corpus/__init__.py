"""Getting a paper onto disk in the shape the pipeline reads. All of it is an input.

Nothing here runs during an extraction. A run reads `data/corpus/<id>/` and never writes it,
so a paper's text and its stage-1 parse are the same on the tenth run as on the first --
which is what makes two runs comparable at all.

    select        which papers are worth extracting
    sync          copy them from the ns-pond corpus
    rebuild       rebuild the text with the tables inlined, and prove it reproduces the corpus
    tables        stage 1: split each coordinate table into the analyses it reports
    abbreviations mine every paper's abbreviation definitions into one store

`tables` is the one step here that costs money -- one model call per table -- and the one the
pipeline treats as load-bearing input rather than regenerating.
"""
