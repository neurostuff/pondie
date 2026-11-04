# pondie-astra

The paper prompt and compact record format of the astra-distill models (Qwen3.8-27B `r3-27b-full`,
NuExtract3 `ev1-full`), and the frozen extraction-record schema they were trained on. The scripts in
`experiments/astra-distill/vista` import it; ns-pond pins it to build the same requests and restore
the same records:

    pondie-astra @ git+https://github.com/neurostuff/pondie.git@<sha>#subdirectory=packages/pondie-astra
