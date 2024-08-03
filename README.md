# Example script to run

```
OPENAI_API_KEY=<FW_API_KEY/TG_API_KEY> python run_meta_benchmarks.py --model-size 8b --provider fw --output-dir gsm8k/fw_3p1_8b/ --eval-set evals__gsm8k__details

# Note - if this crashes due to rate limit/something else, you can rerun the same command to continue - all the previous requests are persisted

python analyze_answers.py --task evals__gsm8k__details --response-path gsm8k/fw_3p1_8b/

> Accuracy: 0.8529188779378317 evals__gsm8k__details gsm8k/fw_3p1_8b/
```

Tasks supported so far are evals__mmlu__details, evals__mmlu__0_shot__cot__details, evals__gsm8k__details, evals__mmlu_pro__details.

Note - we don't know the exact answer extraction logic Meta uses so we rolled out own. Discrepencies may be a result of this.