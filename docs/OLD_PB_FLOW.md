# The old `pb-flow` script

> This script is not being actively maintained. Its functionality has been migrated to its Python reimplementation, `pb-flow.py`.

[pb-flow](scripts/pb-flow) provides a CLI utility to test Phism with Polybench examples. You can grab a rough idea about the whole Phism pipeline over there. You can use `pb-flow` in the following ways:

```sh
./scripts/pb-flow example/polybench       # Run all polybench synth-only, w/o Polyhedral optimization.
./scripts/pb-flow example/polybench -p    # Run all polybench synth-only, w/ Polyhedral optimization.
./scripts/pb-flow example/polybench -c    # Run all polybench w/ cosim, w/ Polyhedral optimization.
./scripts/pb-flow example/polybench -pc   # Run all polybench w/ cosim, w/o Polyhedral optimization.
```

If you attach `-d`, the build effort won't be set to `high`. This can save some time.
