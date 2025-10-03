# OWASP Top 10 LLM QA System

A command-line question-answering system for the OWASP Top 10 for LLM Applications document.

## Prerequisites

1. ***Python 3.12***
2. ***uv***
3. **Ollama**
    - Download from https://ollama.ai
    - After installing, pull the required model:
```bash
ollama pull llama3.2:1b
```
## Setup
**Install Python dependencies with uv:**
```bash
uv sync
```

## Usage
**Run the script:**
```bash
uv run owasp_qa.py 
```

### The system will:
- Load or build a vector database on first run
- Start an interactive QA session
- Type your question and press Enter
- Type `quit`, `q`, or `exit` to stop

or

```
uv run jupyter notebook
```
And run all the cells in the notebook


## Sample questions and answers:
1. 
    #### Question: 
        What risks are associated with insecure output handling?
    #### Answer: 
        The following risks are associated with insecure output handling:  1. **Remote
        Code Execution**: LLM outputs can be used to execute malicious code in the
        target environment. 2. **Cross-Site Scripting (XSS)**: LLM-generated content can
        be used to inject malicious JavaScript or Markdown code into a user's browser.
        3. **SQL Injection**: LLM-generated SQL queries can be executed without proper
        parameterization, allowing attackers to inject malicious SQL code.  These risks
        can lead to various types of attacks, including privilege escalation, remote
        code execution, and data breaches.
2.
    #### Question:
        What is data poisoning?
    #### Answer:
        Data poisoning refers to the intentional manipulation of pre-training, fine-
        tuning, or embedding data to introduce vulnerabilities, backdoors, or biases in
        a machine learning model. This can compromise model security, performance, or
        ethical behavior, leading to harmful outputs or impaired capabilities.
3.
    #### Question:
        How to mitigate supply chain vulnerabilities
    #### Answer:
        To mitigate supply chain vulnerabilities for LLM applications, you can follow
        the steps outlined in the threat model:  1. Carefully vet data sources and
        suppliers, including T&Cs and privacy policies, using trusted suppliers. 2.
        Regularly review and audit supplier Security and Access to ensure no changes in
        their security posture or T&Cs.  Additionally, apply the OWASP Top Ten's
        "A06:2021 – Vulnerable and Outdated Components" mitigations, such as
        vulnerability scanning, management, and patching components, especially for
        development environments with access to sensitive data.