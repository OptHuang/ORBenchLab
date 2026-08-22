# Integration template

Starting point for a third benchmark integration.

```bash
cp templates/integration_template/integration.py src/orbenchlab/integrations/yourbench.py
```

Then follow [`../../docs/adding-a-benchmark.md`](../../docs/adding-a-benchmark.md).

The template is a real module shape with TODOs, not pseudocode: the imports,
the report structure and the check vocabulary are what the shipped integrations
use. Two worked examples sit in `src/orbenchlab/integrations/`, deliberately of
opposite form — `oragentbench.py` (Harbor-native) and `frontieror.py` (official
external harness).

Delete every TODO before opening a pull request. A template comment left in a
real integration is a claim about what was checked that nobody checked.
