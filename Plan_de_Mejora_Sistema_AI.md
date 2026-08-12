# System Instructions / Context: AI Continuous Learning & Improvement Framework

## 1. Executive Overview & System Architecture

This document serves as the foundational guidelines, context, and specification for the AI engine. The goal of this system is to transition from a **static inference model** (where user feedback is logged but unused) to an **active continuous-learning model** (where user corrections directly, dynamically, and measurably update future inference, system prompts, and field-validation rules).

### Current State vs. Target State

| Component | Current State | Target State |
| :--- | :--- | :--- |
| **Analysis Storage** | ✅ Fully Functional | Stores detailed input/output logs alongside user tags. |
| **Feedback Logging** | ✅ Fully Functional | Captures explicit user corrections (`✗ IA Incorrecta`). |
| **`learned_patterns` Table** | ⚠️ Passive Storage | Dynamically queried and injected as few-shot rules before each inference. |
| **Model Retraining (`train.py`)** | ⚠️ Manual Execution | Automated trigger executed upon receiving fresh user corrections. |
| **Metrics & Analytics (`/ai/stats`)** | ⚠️ Unused Endpoint | Powers real-time extension UI dashboards and accuracy telemetry. |

---

## 2. Identified Bottlenecks & System Vulnerabilities

1. **Passive Feedback Loop:** Receiving an `IA Incorrecta` signal stores the correction row but does not modify the weights, prompts, or standard operational evaluation of future requests.
2. **Invisible Metrics:** Absence of feedback UI leaves users and operators with zero visibility into model performance, precision trends, or field error distributions.
3. **Manual Training Overhead:** The process required to refresh fine-tuning weights or rule sets (`train.py`) requires manual intervention instead of event-driven execution.

---

## 3. Implementation Roadmap (Phased Approach)

### Phase 1: Learning Visibility & Telemetry
- **Extension Learning Dashboard:** Display real-time aggregated metrics:
  - Total analyses performed.
  - Approval vs. Rejection breakdown (%).
  - Total correction events logged.
  - Active learned pattern list.
- **Enhanced `/ai/stats` API Endpoint:**
  - Standard JSON response format detailing overall precision, top error-producing fields, and recent correction history logs.

### Phase 2: Dynamic Learning & Active Ingestion
- **Event-Driven Retraining Trigger:**
  - Upon receiving payload on `/ai/feedback`, enqueue or execute an asynchronous update task to refresh `learned_patterns`.
- **Pre-Inference Pattern Injection:**
  - Before invoking the core AI engine, fetch active patterns associated with the current document type or jurisdiction.
  - Inject these patterns as high-priority constraints inside the system prompt (e.g., higher confidence weights for known compound surnames or local ID formats).

### Phase 3: Prompt Engineering & Localization
- **Adaptive System Prompts:**
  - System dynamically updates prompt template files (`prompts/*.txt`) based on recurrent failure patterns.
- **Jurisdiction & Domain Specificity:**
  - Incorporate localized rules (e.g., verifying Paraguay's "CÉDULA DE IDENTIDAD POLICIAL", specific tax ID structures, and naming conventions).

### Phase 4: Precision Telemetry & Reporting
- Generate automated performance digests tracking precision percentage improvements over time, top 5 error fields, and pre- vs. post-correction accuracy metrics.

---

## 4. Extended AI System Specification & Guidelines

### 4.1 System Prompt Architecture & Dynamic Ingestion Scheme
When processing an incoming analysis request, the engine must construct the system prompt dynamically using three layers:

1. **Base System Prompt:** Core structural rules and JSON schemas.
2. **Domain/Locale Rules:** Jurisdiction-specific validation logic (e.g., Paraguay CI formatting, local address syntax).
3. **Dynamic Pattern Injection (From `learned_patterns`):** Top $N$ active patterns derived from explicit user corrections.

```
+-------------------------------------------------------------+
| System Layer 1: Base Analysis System Prompt                 |
+-------------------------------------------------------------+
| System Layer 2: Locale Rules (Paraguay / LATAM Specifics)   |
+-------------------------------------------------------------+
| System Layer 3: Dynamic Injected Patterns (Learned Rules)   |
|  - "BANILDO" is a valid given name; do not flag as typo.    |
|  - "MORALES FERNANDEZ" is a compound family surname.        |
+-------------------------------------------------------------+
                            |
                            v
              +----------------------------+
              | User Payload / Document AI |
              +----------------------------+
```

### 4.2 Feedback Trigger & Pattern Extraction Logic
Whenever a user submits negative feedback (`✗ IA Incorrecta`), the following workflow must execute:

```python
# Conceptual Workflow for Active Feedback Processing

def process_user_feedback(analysis_id, expected_data, actual_data):
    # 1. Log raw feedback entry
    db.save_feedback(analysis_id, expected_data, actual_data)
    
    # 2. Extract specific diffs and identify error patterns
    field_diffs = compute_field_diffs(expected_data, actual_data)
    
    # 3. Insert or update pattern in learned_patterns table
    for field, correction in field_diffs.items():
        db.upsert_learned_pattern(
            field_name=field,
            pattern_type=correction['type'],
            rule_description=correction['rule'],
            confidence_score=0.85
        )
        
    # 4. Trigger asynchronous prompt/model optimization
    trigger_async_retraining()
```

---

## 5. System Target Telemetry Mockup

Upon successful implementation of Phase 1 through Phase 4, the AI system state can be monitored via the following dashboard format:

```text
===================================================================
                   📊 AI LEARNING DASHBOARD
===================================================================
 Total Analyses Conducted:   45
 Approved Analyses:          32 (71.1%)
 Rejected Analyses:          13 (28.9%)
 Explicit Corrections:       8
 Historic Model Accuracy:    65.0% ➔ 72.0% (+7.0%)
 Active Learned Patterns:    12
-------------------------------------------------------------------
 Top Error Fields:
   1. Surname Parsing (Compound Names) -------- [38% of errors]
   2. Local ID / Document Type Recognition ---- [25% of errors]
   3. OCR Noise in Given Names ---------------- [15% of errors]
-------------------------------------------------------------------
 Active Injected Patterns:
   - "BANILDO" is recognized as a valid first name.
   - "MORALES FERNANDEZ" recognized as compound surname.
   - "CÉDULA DE IDENTIDAD POLICIAL" recognized as valid ID title (PY).
===================================================================
```

---

## 6. Implementation Checklist for AI Agent

- [ ] Implement query hook in analysis service to pull active records from `learned_patterns`.
- [ ] Add dynamic text template renderer for injecting patterns into LLM system prompts.
- [ ] Update `/ai/feedback` route to auto-enqueue background rule refinement task (`train.py` / pattern extractor).
- [ ] Expand `/ai/stats` response schema to yield `accuracy_trend`, `top_error_fields`, and `active_patterns`.
- [ ] Render telemetry widgets within the client extension matching the metric specifications.
