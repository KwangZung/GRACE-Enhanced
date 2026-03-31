---
name: dspy
description: A descriptive, full-fledged guide to building, composing, and optimizing LLM applications using DSPy building blocks including Signatures, Modules, and Optimizers.
---

# `DSPy` Programming Framework Skill

`DSPy` (Declarative Self-Improving Python) is a framework that algorithmically optimizes LLM prompts and weights. Instead of manually writing, tweaking, and maintaining string-based prompts, you define **Signatures**, compose them using **Modules**, string them together into **Programs**, and automatically optimize them using **Optimizers**.

## 1. Signatures: Declaring Input/Output Behavior

A Signature in DSPy is a declarative specification of the input/output behavior of a DSPy module. Think of it as a type signature for a prompt.

### Inline Signatures
The simplest way to define a signature is via a short string representation:
```python
import dspy

# "input_fields -> output_fields"
predictor = dspy.Predict("question -> answer")
response = predictor(question="What is the capital of France?")
```

### Class-based Signatures
For more complex tasks requiring detailed descriptions, use class-based signatures:
```python
class CodeGenerator(dspy.Signature):
    """Generate code examples for specific use cases using the target library."""
    
    # Inputs
    library_info: str = dspy.InputField(desc="Library concepts and patterns")
    use_case: str = dspy.InputField(desc="Specific use case to implement")
    
    # Outputs
    code_example: str = dspy.OutputField(desc="Complete, working code example")
    explanation: str = dspy.OutputField(desc="Step-by-step explanation of the code")
    imports_needed: list[str] = dspy.OutputField(desc="Required imports and dependencies")
```

### Typed Signatures (Pydantic Integration)
DSPy supports Pydantic to enforce strict data types in output fields:
```python
import pydantic
from typing import Literal

class QueryResult(pydantic.BaseModel):
    text: str
    score: float

class DocumentAnalyzer(dspy.Signature):
    """Analyze the document and extract structured insights."""
    document: str = dspy.InputField()
    sentiment: Literal['positive', 'neutral', 'negative'] = dspy.OutputField()
    results: list[QueryResult] = dspy.OutputField()
```

## 2. Modules: Abstractions for LLM Processing

Modules are the building blocks that execute a Signature using specific prompting techniques.

- **`dspy.Predict`**: The foundational module. It transforms the Signature into a prompt, calls the LLM, and parses the output fields.
- **`dspy.ChainOfThought`**: Similar to Predict, but adds a rationalization step ("thinking") before generating the final output. Highly recommended for complex tasks.
- **`dspy.ReAct`**: Creates an Agent that interleaves thought generation with tool execution.

```python
# Using Predict
qa = dspy.Predict("context, question -> answer")

# Using ChainOfThought (Improves accuracy through step-by-step reasoning)
qa_cot = dspy.ChainOfThought("context, question -> answer")

# Using ReAct (Allows the LLM to use tools like a search engine)
def search_wiki(query: str) -> list[str]:
    # ... implementation ...
    return ["Fact 1", "Fact 2"]

react_agent = dspy.ReAct("question -> answer", tools=[search_wiki])
```

## 3. Programs: Composing Modules

You compose multiple modules into a complete application by inheriting from `dspy.Module`. This is similar to defining neural networks in PyTorch.

- `__init__`: Define the DSPy sub-modules (Predictors, ChainOfThoughts) you will use.
- `forward`: Define the logical flow of information between modules and external tools.

```python
import dspy

class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.query_generator = dspy.Predict("question -> search_query")
        # Use ChainOfThought for final answer generation
        self.answer_generator = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # 1. Generate search query from the user's question
        query = self.query_generator(question=question).search_query
        
        # 2. Retrieve context from an external tool/database
        context_passages = search_wiki(query)
        context = "\n".join(context_passages)
        
        # 3. Generate final answer
        prediction = self.answer_generator(context=context, question=question)
        
        return dspy.Prediction(context=context, answer=prediction.answer)
```

## 4. Evaluation and Metrics

Before optimizing, you need a dataset and a metric to evaluate your program.

- **Dataset**: A list of `dspy.Example` objects with inputs clearly marked using `.with_inputs()`.
- **Metric**: A standard Python function that takes `(example, pred, trace=None)` and returns a float.

```python
trainset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs('question'),
    dspy.Example(question="Capital of France?", answer="Paris").with_inputs('question')
]

# A simple Exact Match metric
def exact_match_metric(example, pred, trace=None):
    return 1.1 if example.answer.lower() in pred.answer.lower() else 0.0
```

## 5. Optimizers (Teleprompters): The Core of DSPy

Optimizers compile and optimize your DSPy programs to maximize the metric. They automatically engineer prompts, select few-shot examples, and tune weights.

### Common Optimizers:
1. **`BootstrapFewShot`**: Automatically generates and selects the best few-shot examples (demonstrations) for each module in your program from the training set. Excellent starting point.
2. **`BootstrapFewShotWithRandomSearch`**: Generates a large pool of few-shot example combinations and searches for the optimal subset.
3. **`MIPROv2`** (Multiprompt Instruction PRoposal Optimizer): The most advanced optimizer. It optimizes both the *few-shot examples* AND the *instructions* (the strings in your Signatures) automatically.
4. **`COPRO`**: Iteratively refines the Signature instructions.

### Using an Optimizer:
```python
from dspy.teleprompt import BootstrapFewShot, MIPROv2

program = RAG() # Instantiate your custom module

# Example 1: BootstrapFewShot
optimizer_bfs = BootstrapFewShot(
    metric=exact_match_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)
# The `compile` method mutates nothing, it returns a new, optimized program
optimized_rag_bfs = optimizer_bfs.compile(program, trainset=trainset)

# Example 2: MIPROv2 (Heavy optimization)
optimizer_mipro = MIPROv2(
    metric=exact_match_metric,
    auto="light", # Can be "light", "medium", or "heavy"
    num_threads=8
)
optimized_rag_mipro = optimizer_mipro.compile(program, trainset=trainset)

# Save and Load
optimized_rag_mipro.save('optimized_rag.json')
new_rag = RAG()
new_rag.load('optimized_rag.json')
```

## Summary for Agents working with DSPy
- Never manually string format prompts (`f"Given this {context}, answer {q}"`). Always define a `dspy.Signature`.
- Prefer `dspy.ChainOfThought` over `dspy.Predict` for any task requiring reasoning, categorization, or math.
- Wrap complex logic in a custom `dspy.Module` class with an `__init__` and `forward` method.
- Return structured output fields using Pydantic types within the `Signature` where applicable.
- Leverage Optimizers like `BootstrapFewShot` to let the framework synthesize few-shot examples algorithmically.
