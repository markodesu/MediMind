# How the ML Backend Works: Complete Flow Explanation

This document explains step-by-step what happens when a user sends a message like "i have headache" through the MediMind backend.

## Overview: The Complete Journey

```
User Input → FastAPI Router → Safety Check → Model Loading → Prompt Formatting → 
Tokenization → Model Inference → Response Decoding → Post-Processing → 
Confidence Scoring → API Response
```

---

## Step-by-Step: "i have headache"

### **Step 1: Request Arrives at FastAPI Router**
**File:** `backend/app/routers/chat.py`

```python
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_message = request.message  # "i have headache"
    history = request.history or []  # Empty for first message
```

**What happens:**
- Frontend sends POST request to `/api/v1/chat`
- Request body: `{"message": "i have headache", "history": []}`
- FastAPI validates using `ChatRequest` schema (Pydantic)
- Extracts `user_message = "i have headache"` and `history = []`

---

### **Step 2: Urgent Case Safety Check**
**File:** `backend/app/routers/chat.py` (line 33)
**Function:** `should_redirect_to_doctor(user_message)`

```python
if should_redirect_to_doctor(user_message):
    return ChatResponse(...)  # Redirect to doctor
```

**What happens:**
- Checks for urgent keywords: "severe", "emergency", "chest pain", "bleeding", etc.
- Checks for diagnostic patterns: "do i have...", "am i...", "prescribe", etc.
- For "i have headache": **No urgent keywords found** → Returns `False`
- **Result:** Continue to model inference

**Code location:** `backend/app/services/llm.py` lines 190-248

---

### **Step 3: Call LLM Service**
**File:** `backend/app/routers/chat.py` (line 41)

```python
response, confidence = generate_response(user_message, history)
```

**Function:** `generate_response(message, history)` in `backend/app/services/llm.py`

---

### **Step 4: Model Loading (Lazy Loading)**
**File:** `backend/app/services/llm.py` (lines 536-537)

```python
if tokenizer is None or model is None:
    load_model()
```

**What happens:**
- **First request:** Model is loaded from HuggingFace (`microsoft/phi-2`)
- **Subsequent requests:** Model is already in memory (fast!)
- Model loaded on GPU if available, otherwise CPU
- Optional: LoRA adapter loaded if `LORA_MODEL_PATH` is set in `.env`

**Model details:**
- **Base Model:** `microsoft/phi-2` (2.7B parameters)
- **Tokenizer:** AutoTokenizer (converts text ↔ tokens)
- **Device:** CUDA (GPU) if available, else CPU

**Code location:** `backend/app/services/llm.py` lines 133-187

---

### **Step 5: Determine Question Complexity**
**File:** `backend/app/services/llm.py` (line 540)

```python
is_complex = is_complex_question(message, history)
```

**What happens:**
- Checks if question is complex (needs longer response)
- For "i have headache": **Simple question** → `is_complex = False`
- Complex questions get 2x token limit (360 vs 180 tokens)

**Complexity indicators:**
- Multiple symptoms mentioned
- Long question (>50 words)
- Contains temporal references with history
- Medical terminology

**Code location:** `backend/app/services/llm.py` lines 249-288

---

### **Step 6: Format Conversation Prompt**
**File:** `backend/app/services/llm.py` (line 543)

```python
formatted_prompt = format_conversation(message, history)
```

**What happens:**
- Builds the complete prompt for the model
- Includes system instructions, conversation history, and current message

**For "i have headache" (first message, simple question):**

```
You are MediMind, a friendly health information assistant for university students.

IMPORTANT: Always respond directly to the user's CURRENT question. Read their message carefully.

If the user asks "what is [something]" (like "what is flu"), explain what it is in 2-3 sentences.
If the user asks "what to do" or "what should I do", give practical self-care steps.
If the user describes a symptom (like "my head hurts"), acknowledge it and ask 1-2 follow-up questions.

Examples:
Human: what is flu
Assistant: Flu (influenza) is a viral infection that affects your respiratory system. It typically causes fever, body aches, cough, fatigue, and sometimes chills. Most people recover with rest and fluids.

Human: i have flu what to do
Assistant: If you have flu, rest at home, drink plenty of fluids like water and warm tea, and get enough sleep. You can use simple pain relievers like paracetamol for fever and body aches (check with a pharmacist first). If symptoms get worse, see a doctor.

Human: my head hurts
Assistant: I'm sorry to hear that. When did the headache start, and how strong is it (mild, moderate, or strong)? Also, do you have any other symptoms like fever or nausea?

Do not diagnose or prescribe. Be kind and helpful.

If the user asks for contact information for Dr. Kyal or the campus doctor, provide: Dr. Kyal, phone: +996708136013, location: 1st floor, Academic Block, near GYM.

Human: i have headache
Assistant:
```

**Key components:**
1. **System instruction** (adapts to simple/complex)
2. **Contact info** (dynamically injected from settings)
3. **Conversation history** (if any - empty for first message)
4. **Current message** with "Human:" prefix
5. **"Assistant:"** marker (where model should start generating)

**Code location:** `backend/app/services/llm.py` lines 291-390

---

### **Step 7: Tokenization**
**File:** `backend/app/services/llm.py` (line 549)

```python
inputs = tokenizer(formatted_prompt, return_tensors="pt")
```

**What happens:**
- Converts text prompt into token IDs (numbers the model understands)
- Example: "i have headache" → `[123, 456, 7890]` (token IDs)
- Returns PyTorch tensors: `{"input_ids": tensor([...]), "attention_mask": tensor([...])}`
- Moves tensors to GPU if available

**Token details:**
- Uses model's vocabulary (50,000+ tokens for phi-2)
- Special tokens: `<|endoftext|>`, padding tokens
- Max length handled automatically

---

### **Step 8: Model Inference (Text Generation)**
**File:** `backend/app/services/llm.py` (lines 556-563)

```python
with torch.no_grad():  # Disable gradient computation (faster inference)
    outputs = model.generate(
        **inputs,
        max_new_tokens=180,  # Simple question = 180 tokens
        do_sample=False,    # Greedy decoding (deterministic)
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,  # Reduce repetition
    )
```

**What happens:**
- Model processes input tokens through transformer layers
- Generates new tokens one by one (autoregressive generation)
- Stops when:
  - `max_new_tokens` reached (180 tokens)
  - EOS token generated (`<|endoftext|>`)
  - Model decides to stop

**Generation process:**
1. Model predicts next token probability distribution
2. Greedy decoding: picks highest probability token
3. Adds token to sequence
4. Repeats until stop condition

**Output:** Tensor of token IDs (input + generated tokens)

**Example output tokens:**
```
[..., 123, 456, 7890, 234, 567, 890, 345, ...]
 ↑ input tokens    ↑ generated tokens
```

---

### **Step 9: Decode Response**
**File:** `backend/app/services/llm.py` (lines 566-591)

```python
full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**What happens:**
- Converts token IDs back to text
- Removes special tokens
- Extracts only the assistant's response (after "Assistant:")

**Example:**
```
Full decoded text:
"You are MediMind... Human: i have headache Assistant: I'm sorry to hear that. When did the headache start, and how strong is it (mild, moderate, or strong)? Also, do you have any other symptoms like fever or nausea?"

Extracted response:
"I'm sorry to hear that. When did the headache start, and how strong is it (mild, moderate, or strong)? Also, do you have any other symptoms like fever or nausea?"
```

**Code handles:**
- Splitting on "Assistant:" marker
- Removing "Human:" if it appears in response
- Fallback extraction if format differs

---

### **Step 10: Post-Processing**
**File:** `backend/app/services/llm.py` (line 594)

```python
text = post_process_response(raw_text, message, is_complex)
```

**What happens:**
- **Safety filters:**
  - Removes explicit dosages: "50mg" → removed
  - Replaces prescription drugs: "ibuprofen" → "medicine"
  - Removes diagnostic language: "prescribe" → "assess"

- **Quality improvements:**
  - Extracts complete sentences
  - Limits length (200 chars for simple, 400 for complex)
  - Ensures proper punctuation
  - Selects most relevant sentences

**For our example:**
```
Input: "I'm sorry to hear that. When did the headache start, and how strong is it (mild, moderate, or strong)? Also, do you have any other symptoms like fever or nausea?"

Output: "I'm sorry to hear that. When did the headache start, and how strong is it (mild, moderate, or strong)? Also, do you have any other symptoms like fever or nausea?"
```
(No changes needed - already clean)

**Code location:** `backend/app/services/llm.py` lines 393-527

---

### **Step 11: Calculate Confidence Score**
**File:** `backend/app/services/llm.py` (lines 596-687)

**Multi-factor confidence algorithm:**

#### **11.1 Base Confidence (Response Length)**
```python
response_length = 95  # Characters
if response_length < 20:
    confidence = 0.5
elif response_length < 50:
    confidence = 0.55 + ((95 - 20) / 30) * 0.15  # = 0.925 (capped)
elif response_length < 150:
    confidence = 0.7 + ((95 - 50) / 100) * 0.15  # = 0.7675
```
**Result:** `confidence = 0.7675` (good length)

#### **11.2 User Message Analysis**
```python
user_message_lower = "i have headache"
has_medical_keywords = True  # "headache" is in medical keywords list
has_dangerous_keywords = False  # No dangerous keywords
is_info_question = False
is_what_to_do = False
```

**Adjustments:**
- Medical keywords found → No penalty
- No dangerous keywords → No penalty
- **Confidence remains:** `0.7675`

#### **11.3 Response Content Analysis**
```python
response_lower = "i'm sorry to hear that. when did the headache start..."
has_medical_context = True  # Contains "headache"
```

**Adjustments:**
- Medical context found → `confidence += 0.1`
- **New confidence:** `0.8675`

#### **11.4 Quality Checks**
```python
has_punctuation = True  # Has "?" and "."
# No repetition detected
# No code detected
```

**Adjustments:**
- Complete sentences → No penalty
- **Final confidence:** `0.8675`

#### **11.5 Normalize**
```python
confidence = max(0.3, min(0.8675, 0.85))  # Cap at 0.85
confidence = 0.85
```

**Final confidence:** `0.85` (85%)

**Code location:** `backend/app/services/llm.py` lines 596-687

---

### **Step 12: Return API Response**
**File:** `backend/app/routers/chat.py` (lines 46-50)

```python
return ChatResponse(
    answer="I'm sorry to hear that. When did the headache start, and how strong is it (mild, moderate, or strong)? Also, do you have any other symptoms like fever or nausea?",
    confidence=0.85,
    safe=0.85 >= 0.3  # True
)
```

**Final JSON response:**
```json
{
  "answer": "I'm sorry to hear that. When did the headache start, and how strong is it (mild, moderate, or strong)? Also, do you have any other symptoms like fever or nausea?",
  "confidence": 0.85,
  "safe": true
}
```

---

## Key Components Summary

### **1. Safety Layers**
- **Pre-filtering:** Urgent keywords → immediate redirect
- **Model instructions:** Explicit "do not diagnose" prompts
- **Post-processing:** Removes dosages and prescription language
- **Confidence scoring:** Penalizes dangerous keywords

### **2. Model Architecture**
- **Base:** microsoft/phi-2 (2.7B parameters)
- **Optional:** LoRA fine-tuning adapter
- **Inference:** Greedy decoding (deterministic)
- **Device:** GPU if available, else CPU

### **3. Prompt Engineering**
- **System instructions:** Adapt to simple/complex questions
- **Few-shot examples:** Shows model desired format
- **Context handling:** Includes conversation history
- **Dynamic injection:** Contact info from settings

### **4. Confidence Scoring**
- **Multi-factor:** Length, medical relevance, danger detection, quality
- **Range:** 0.3 - 0.85 (normalized)
- **Purpose:** Determines `safe` flag (threshold = 0.3)
- **Not used for routing:** Urgent cases handled separately

---

## Example: With Conversation History

**Previous conversation:**
```
User: "I have a headache"
Assistant: "I'm sorry to hear that. When did the headache start..."
User: "It still hurts"
```

**What happens:**
1. History included in prompt formatting
2. Model sees temporal reference "it still hurts"
3. Model understands "it" = headache from history
4. Generates context-aware response about the continuing headache

**Prompt includes:**
```
Previous conversation (use this to understand temporal references...):
Human: I have a headache
Assistant: I'm sorry to hear that. When did the headache start...
Human: It still hurts
Assistant:
```

---

## Performance Considerations

- **First request:** ~3-5 seconds (model loading)
- **Subsequent requests:** ~0.5-2 seconds (inference only)
- **GPU vs CPU:** GPU is 5-10x faster
- **Token limit:** 180 tokens (simple) = ~1-2 sentences
- **Memory:** ~6GB GPU RAM for phi-2

---

## Error Handling

- **Model not loaded:** Auto-loads on first request
- **Empty response:** Returns fallback message
- **Invalid input:** Pydantic validation catches it
- **Generation failure:** Returns error message with doctor contact

---

This completes the full journey of a user message through the ML backend! 🎯

