# MediMind: Complete Process Explanation

This document provides a comprehensive explanation of how MediMind works, from user input to AI response, including model selection, dataset preparation, confidence scoring, and the challenges we overcame.

---

## Table of Contents

1. [End-to-End Process Flow](#end-to-end-process-flow)
2. [Model Selection: Why microsoft/phi-2?](#model-selection-why-microsoftphi-2)
3. [Dataset Preparation and Cleaning](#dataset-preparation-and-cleaning)
4. [Confidence Scoring System](#confidence-scoring-system)
5. [Challenges and Solutions](#challenges-and-solutions)

---

## End-to-End Process Flow

### Step-by-Step: What Happens When a User Sends a Message

```
User Input → Frontend → Backend Router → Safety Checks → LLM Service → Response Generation → Confidence Scoring → Response Routing → User
```

#### 1. **Frontend (React + TypeScript)**
- User types a message in the chat interface
- Frontend sends HTTP POST request to `/api/v1/chat` endpoint
- Includes conversation history for context-aware responses

#### 2. **Backend Router (`app/routers/chat.py`)**
- Receives `ChatRequest` with `message` and `history`
- First safety check: `should_redirect_to_doctor()`
  - Detects urgent symptoms (bleeding, chest pain, etc.)
  - Returns immediate doctor referral if detected

#### 3. **LLM Service (`app/services/llm.py`)**

**3.1 Static Response Check**
- Checks if message matches common patterns (headache, cold, flu, etc.)
- Returns predefined, safe responses for common queries
- **Why?** Reduces latency, ensures consistency, and avoids unnecessary LLM calls

**3.2 Safety Redirect Check**
- Second safety layer: checks for dangerous keywords
- Returns doctor referral with low confidence (0.3)

**3.3 Model Loading (Lazy Loading)**
- If model not loaded, loads `microsoft/phi-2` model and tokenizer
- Uses GPU if available, falls back to CPU
- Model is cached in memory for subsequent requests

**3.4 Prompt Formatting**
- `format_conversation()` builds the prompt:
  - System instructions (adapts based on question complexity)
  - Conversation history (last 6 messages for context)
  - Current user message
  - Explicit instruction: "Always respond to CURRENT message, not previous topics"

**3.5 Question Complexity Detection**
- `is_complex_question()` analyzes message:
  - Multiple questions
  - Long questions (>15 words)
  - Complex keywords ("explain", "describe", "why", "how")
  - Follow-up questions with context words

**3.6 Tokenization**
- Input prompt is tokenized using phi-2 tokenizer
- Moved to GPU if available

**3.7 Generation**
- Model generates response using greedy decoding (deterministic)
- Token limit: 180 tokens (simple) or 360 tokens (complex)
- Extracts only the assistant's response from full output

**3.8 Post-Processing**
- `post_process_response()`:
  - Ensures complete sentences
  - Removes incomplete thoughts
  - Truncates if too long
  - Maintains natural flow

**3.9 Confidence Scoring**
- Multi-factor confidence calculation (detailed below)
- Returns response text and confidence score (0.0-1.0)

#### 4. **Response Routing (`app/routers/chat.py`)**
- If confidence < 0.6 (threshold): Routes to Dr. Kyal
- If confidence >= 0.6: Returns AI response
- Response includes: `answer`, `confidence`, `safe` flag

#### 5. **Frontend Display**
- Displays response to user
- Shows confidence indicator (if implemented)
- Maintains conversation history for next message

---

## Model Selection: Why microsoft/phi-2?

### Model Comparison

| Model | Parameters | Size | Speed | Quality | Our Choice |
|-------|------------|------|-------|---------|------------|
| **phi-2** | 2.7B | ~5GB | Fast | Good | ✅ **Selected** |
| GPT-3.5 | 175B | ~350GB | Slow | Excellent | ❌ Too large |
| GPT-4 | ~1T | ~700GB | Very slow | Best | ❌ Too large, expensive |
| LLaMA-2 | 7B-70B | 14-140GB | Medium | Good | ❌ Too large for local |
| Mistral | 7B | 14GB | Medium | Good | ❌ Larger than needed |

### Why We Chose phi-2

1. **Size Efficiency**
   - Only 2.7B parameters (~5GB)
   - Runs on consumer GPUs (8GB VRAM) or CPU
   - Fast inference (~1-3 seconds per response)

2. **Performance**
   - Trained on high-quality synthetic data
   - Strong reasoning capabilities despite small size
   - Good balance between speed and quality

3. **Open Source**
   - Free to use (MIT license)
   - No API costs or rate limits
   - Full control over deployment

4. **Fine-Tuning Support**
   - Compatible with LoRA (Low-Rank Adaptation)
   - Efficient fine-tuning on medical data
   - Maintains base model capabilities

5. **Medical Domain Suitability**
   - General-purpose model that adapts well to medical Q&A
   - With fine-tuning, provides accurate medical guidance
   - Safety-conscious responses

### Model Architecture

- **Type**: Causal Language Model (Decoder-only)
- **Architecture**: Transformer-based
- **Training**: Pre-trained on diverse text data
- **Fine-tuning**: Optional LoRA adapter for medical domain

### Training Data

**Base Model (phi-2):**
- Trained on synthetic data (high-quality, diverse)
- Includes code, math, reasoning, and general knowledge
- Not specifically medical-focused

**Our Fine-Tuning (Optional):**
- Medical Q&A datasets (MEDIQA, medical_questions_pairs)
- ~10,000+ medical question-answer pairs
- LoRA fine-tuning (parameter-efficient)

---

## Dataset Preparation and Cleaning

### Dataset Sources

1. **MEDIQA 2019 Task 3 (QA)**
   - Medical question-answering dataset
   - XML format with Q&A pairs
   - Multiple answer options per question

2. **medical_questions_pairs (HuggingFace)**
   - Large collection of medical Q&A pairs
   - Various medical topics
   - Pre-formatted JSON

### Dataset Cleaning Process (`app/data/prepare_dataset.py`)

#### Step 1: Data Loading
```python
# Load from multiple sources
- MEDIQA XML files → Parse XML structure
- HuggingFace datasets → Direct loading
```

#### Step 2: Safety Filtering
**Unsafe Content Detection:**
- **Prescription Dosages**: Filters explicit dosages (e.g., "take 500mg", "prescribe 10ml")
- **Pattern Matching**: Uses regex to detect dangerous patterns
- **Very Lenient**: Only blocks specific prescription instructions

**Why Lenient?**
- Trust confidence scoring to handle edge cases
- Allow general medical information
- Safety disclaimers added to all responses

#### Step 3: Text Simplification
- Remove excessive whitespace
- Replace abbreviations (Dr. → doctor)
- Normalize formatting
- Ensure readability

#### Step 4: Safety Disclaimers
- Adds disclaimer to all answers:
  > "If your symptoms worsen or feel concerning, seek help from a healthcare professional."

#### Step 5: Format Transformation
**Input Format:**
```json
{
  "question": "...",
  "answer": "..."
}
```

**Output Format (dataset.jsonl):**
```json
{
  "instruction": "What is a headache?",
  "response": "A headache is pain in the head...",
  "source": "mediqa_qa",
  "safety_level": "safe"
}
```

#### Step 6: Quality Checks
- Minimum answer length: 10 characters
- Maximum answer length: 5000 characters
- Validates question-answer pairs
- Removes duplicates

### Final Dataset Statistics

- **Total Entries**: ~10,000+ Q&A pairs
- **Sources**: MEDIQA 2019, medical_questions_pairs
- **Format**: JSONL (one JSON object per line)
- **Safety Level**: All entries marked as "safe"
- **Coverage**: General health, symptoms, treatments, common conditions

---

## Confidence Scoring System

### Overview

Confidence scoring is a **multi-factor algorithm** that evaluates:
1. Response quality
2. User message relevance
3. Medical context
4. Safety indicators

### Confidence Calculation Flow

```
Base Confidence (from response length)
    ↓
User Message Analysis (medical keywords, danger indicators)
    ↓
Response Content Analysis (medical relevance)
    ↓
Quality Checks (completeness, repetition, code detection)
    ↓
Final Confidence Score (0.2 - 0.75)
```

### Detailed Algorithm

#### 1. **Base Confidence (Response Length)**

```python
if response_length < 15:
    confidence = 0.3  # Too short = low confidence
elif response_length < 50:
    confidence = 0.4 + ((length - 15) / 35) * 0.15  # 0.4-0.55
elif response_length < 150:
    confidence = 0.55 + ((length - 50) / 100) * 0.2  # 0.55-0.75
else:
    confidence = 0.75  # Good length, capped at 0.75
```

**Rationale**: Longer, complete responses indicate better understanding.

#### 2. **User Message Analysis**

**Medical Keywords Detection:**
- Checks if user message contains medical terms
- Keywords: symptom, health, medical, pain, fever, headache, etc.
- **If found**: No penalty
- **If not found**: -0.2 confidence

**Dangerous Keywords Detection:**
- Urgent symptoms: severe, emergency, chest pain, bleeding, etc.
- **If found**: 
  - Multiply confidence by 0.5 (halve it)
  - Additional -0.15 penalty
  - **Result**: Very low confidence → Routes to doctor

**Unknown Patterns:**
- Non-medical patterns: "xyz", "abc", "test", random text
- **If found**: -0.15 additional penalty

#### 3. **Response Content Analysis**

**Medical Context in Response:**
- Checks if response contains medical terms
- **If found**: +0.05 boost
- **If not found**: -0.1 penalty

#### 4. **Quality Checks**

**Complete Sentences:**
- Checks for punctuation (. ! ?)
- **If missing**: Multiply by 0.8 (20% penalty)

**Repetition Detection:**
- Checks if response repeats words excessively
- **If repetitive**: Multiply by 0.7 (30% penalty)

**Code/Unrelated Content:**
- Detects code patterns: "def ", "import ", "function", etc.
- **If found**: Multiply by 0.4 (60% penalty)

#### 5. **Final Normalization**

```python
confidence = max(0.2, min(confidence, 0.75))
```

- Minimum: 0.2 (always some uncertainty)
- Maximum: 0.75 (pre-trained model, not perfect)

### Confidence Threshold

- **Threshold**: 0.6
- **Below 0.6**: Routes to Dr. Kyal
- **Above 0.6**: Returns AI response

### Example Confidence Calculations

**Example 1: Normal Medical Query**
```
User: "I have a headache"
Response: "For a mild headache, rest and drink water..."
Length: 80 chars
Medical keywords: ✅ (headache, pain)
Dangerous keywords: ❌
Confidence: 0.7 → ✅ Returns AI response
```

**Example 2: Dangerous Symptom**
```
User: "I have severe chest pain"
Response: "This sounds serious..."
Length: 50 chars
Medical keywords: ✅
Dangerous keywords: ✅ (severe, chest pain)
Confidence: (0.55 * 0.5) - 0.15 = 0.125 → ❌ Routes to doctor
```

**Example 3: Unknown Query**
```
User: "xyz abc test"
Response: "I'm not fully confident..."
Length: 30 chars
Medical keywords: ❌
Unknown patterns: ✅
Confidence: 0.4 - 0.2 - 0.15 = 0.05 → ❌ Routes to doctor
```

---

## Challenges and Solutions

### Challenge 1: Model Size vs. Performance

**Problem:**
- Needed a model that runs locally (no API costs)
- Must be fast enough for real-time chat
- Must provide quality medical guidance

**Solution:**
- Selected phi-2 (2.7B parameters)
- Implemented lazy loading (load once, reuse)
- Added static responses for common queries (reduces LLM calls)
- Optimized token limits based on complexity

**Result:**
- Fast responses (1-3 seconds)
- Runs on consumer hardware
- Good quality for medical Q&A

---

### Challenge 2: Safety and Medical Accuracy

**Problem:**
- Cannot provide medical diagnoses
- Must avoid prescription dosages
- Need to route serious cases to professionals

**Solution:**
- **Multi-layer safety system:**
  1. Static responses for common queries (pre-validated)
  2. Urgent symptom detection (immediate doctor routing)
  3. Confidence-based routing (low confidence → doctor)
  4. Safety disclaimers in all responses
  5. Dataset filtering (removes prescription dosages)

**Result:**
- Safe responses for common queries
- Automatic routing for serious cases
- Clear disclaimers and doctor contact info

---

### Challenge 3: Context Awareness

**Problem:**
- Model sometimes responded to previous messages instead of current one
- User switches topics (headache → stomachache), but bot keeps talking about headache

**Solution:**
- **Enhanced system instructions:**
  - Explicit: "Always respond to CURRENT message"
  - "If user mentions NEW symptom, respond to NEW topic"
  - Only use history when user explicitly references it
- **History formatting:**
  - Labeled as "Previous conversation (for context only)"
  - Emphasizes current message importance

**Result:**
- Bot correctly responds to current message
- Handles topic switching properly
- Maintains context when needed

---

### Challenge 4: Confidence Scoring Accuracy

**Problem:**
- Initial confidence scoring was too simplistic
- Didn't account for dangerous symptoms
- Didn't detect non-medical queries

**Solution:**
- **Multi-factor confidence algorithm:**
  - Response length analysis
  - User message keyword detection (medical, dangerous, unknown)
  - Response content analysis
  - Quality checks (completeness, repetition, code)
- **Dangerous keyword detection:**
  - Significantly lowers confidence for urgent symptoms
  - Ensures routing to doctor

**Result:**
- More accurate confidence scores
- Better routing decisions
- Handles edge cases (unknown queries, dangerous symptoms)

---

### Challenge 5: Dataset Quality and Cleaning

**Problem:**
- Multiple dataset sources with different formats
- Some entries contain prescription dosages
- Inconsistent quality and formatting

**Solution:**
- **Unified cleaning pipeline:**
  - XML parsing for MEDIQA data
  - JSON parsing for HuggingFace datasets
  - Safety filtering (removes prescription dosages)
  - Text simplification
  - Safety disclaimers added
  - Quality validation (length, completeness)
- **Format standardization:**
  - All entries in JSONL format
  - Consistent field names (instruction, response)

**Result:**
- Clean, safe dataset
- Consistent format
- Ready for fine-tuning

---

### Challenge 6: Response Quality and Consistency

**Problem:**
- Model sometimes generates incomplete responses
- Inconsistent response lengths
- Sometimes includes code or unrelated content

**Solution:**
- **Post-processing pipeline:**
  - Ensures complete sentences
  - Removes incomplete thoughts
  - Truncates if too long
  - Detects and penalizes code/unrelated content
- **Static responses:**
  - Pre-written responses for common queries
  - Ensures consistency and safety
  - Reduces latency

**Result:**
- Consistent, complete responses
- Better user experience
- Faster responses for common queries

---

### Challenge 7: Fine-Tuning Resource Constraints

**Problem:**
- Full fine-tuning requires 16GB+ VRAM
- Most students don't have high-end GPUs
- Training time can be very long

**Solution:**
- **LoRA (Low-Rank Adaptation):**
  - Parameter-efficient fine-tuning
  - Reduces memory by 70-90%
  - Trains faster
  - Works with 4-bit quantization (6-8GB VRAM)
- **Local training:**
  - Train on local GPU (8GB+ VRAM recommended)
  - Pre-configured training scripts
  - Full control over training process

**Result:**
- Fine-tuning accessible on consumer hardware
- Faster training times
- Optional fine-tuning (base model works well without fine-tuning)

---

## Summary

MediMind uses a **sophisticated, multi-layer system** to provide safe, accurate medical guidance:

1. **Efficient Model**: phi-2 (2.7B) for fast, local inference
2. **Safety First**: Multiple safety checks and routing mechanisms
3. **Context Aware**: Responds to current message, maintains history when needed
4. **Smart Confidence**: Multi-factor scoring for accurate routing decisions
5. **Clean Data**: Carefully filtered and standardized medical Q&A dataset
6. **Quality Responses**: Post-processing ensures complete, relevant answers

The system balances **safety, accuracy, and performance** to provide a reliable health guidance chatbot for UCA students and staff.

---

**Last Updated:** 2025-12-03

