SUITES = {
    "prompt_injection": [
        {
            "hf_name": "Smooth-3/llm-prompt-injection-attacks",
            "split": "validation",
            "text_field": "text",
            "label_field": "labels"
        }
    ],

    "security_eval": [
        {
            "hf_name": "Smooth-3/llm-prompt-injection-attacks",
            "split": "validation",
            "text_field": "text",
            "label_field": "labels"
        },
        {
            "hf_name": "lmsys/prompt-injection",
            "split": "test"
        }
    ]
}