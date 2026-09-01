"""What the model is asked, and what the paper looks like when it is asked.

    render      the schema and the paper, rendered into the prompt for one pass
    preprocess  deterministic transforms of the paper text before it is sent

The split matters: `render` decides what is asked for, `preprocess` decides what the model
reads. Measured separately in docs/text-preprocessing-experiments.md, because a transform
that shortens the prompt and moves the answer is not a saving.
"""
