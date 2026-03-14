# docs/adding_a_model.md

## Adding a Model Plugin

**Time required**: ~1 hour for a HuggingFace checkpoint.

### Step 1: Create your plugin file

```bash
cp bench/models/distilbert.py bench/models/my_model.py
```

Edit `my_model.py`:
- Change class name to `MyModelDetector`
- Update `model_name` in `load()`
- Update `metadata` property with accurate values
- Change `@register` ID to something unique

### Step 2: Register in YAML

Add to `configs/model_registry.yaml`:
```yaml
- id: my-model-id
  class: bench.models.my_model.MyModelDetector
  description: Brief description
  paper_url: https://arxiv.org/abs/...
  model_card_url: https://huggingface.co/...
```
### Step 3: Validate

```bash
python scripts/validate_plugin.py --model my-model-id
```

### Step 4: Run benchmark

```bash
python -m bench.run --config configs/quick_smoke_test.yaml
```

### Step 5: Open a PR

Include: plugin file, registry YAML, results artifact, updated leaderboard.md.