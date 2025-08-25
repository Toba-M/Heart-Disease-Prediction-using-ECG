import os
import openai
import warnings
import pandas as pd
import re
import random
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.schema import SystemMessage, HumanMessage
from rapidfuzz import fuzz,process

# ─── Load environment ─────────────────────────────────────────────────────────
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    print("⚠️  No .env file found; will try environment variables directly.")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("🔑  OPENAI_API_KEY not set!")
openai.api_key = api_key
warnings.filterwarnings("ignore")

# ─── LLM Setup ────────────────────────────────────────────────────────────────
llm = ChatOpenAI(api_key=api_key, temperature=0.0, model="gpt-3.5-turbo")

# ─── Agent Tools ──────────────────────────────────────────────────────────────
pubmed_tool = PubmedQueryRun()
arxiv_tool = ArxivQueryRun()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [pubmed_tool, arxiv_tool, wiki_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# ─── Memory ───────────────────────────────────────────────────────────────────
chat_memory = ConversationBufferMemory(
    memory_key="history",  
    input_key="input",
    output_key="output",
    return_messages=True
)

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM = """
You are CardioBot, a friendly and precise assistant focused on heart health and ECG-related concerns.
Always warmly acknowledge greetings like "Hi" or "How are you?" — respond briefly and politely before inviting a health-related question.
You specialize in helping users understand symptoms (e.g., fatigue, chest pain, palpitations), ECG results, heart conditions, and treatment options.
If the user asks for advice on medications or anything relating to medical things, supplements, or treatments, treat it as a medical question.  
— Ask for more context if needed (e.g., symptoms, diagnosis).  
— Never recommend prescription drugs without understanding the user’s condition.  
— Encourage follow-up with a healthcare provider for personal medication guidance.

🔍 When a medical concern is raised:
- Explain the condition in 1–2 clear, friendly sentences.
- Avoid jargon; define any medical terms used.
- Provide 3–5 helpful lifestyle tips or management steps.
- Ask follow-up questions to check for red-flag symptoms.
- If red-flag symptoms appear (e.g., fainting, severe chest pain, sudden breathlessness), advise the user to seek emergency care immediately.

Tone: Always empathetic, concise, and supportive.
"""


# ─── Prompt Chains ────────────────────────────────────────────────────────────
extract_prompt = ChatPromptTemplate.from_template(
    "You are an information extraction system. From the text below, extract ONLY the exact medical diagnosis or condition mentioned, "
    "without adding any explanations, extra words, or descriptions.\n"
    "If the text contains a normal/benign finding like 'normal sinus rhythm', 'sinus rhythm', or 'NSR', return it exactly as written.\n"
    "If multiple conditions are mentioned, return them separated by commas.\n"
    "If no medical condition is found, return ONLY the word None (no quotes).\n"
    "Your answer must never contain any extra sentences or formatting.\n\n"
    "Text: {medical_text}"
)
extract_diag_prompt = ChatPromptTemplate.from_template(
    "You are an information extraction system. From the text below, extract ONLY the exact medical diagnoses or conditions mentioned "
    "Extract them as a JSON array of exactly two strings, preserving their wording.\n"
    "Output must be a JSON array of exactly two strings, and nothing else. For example:"
    "['Diagnosis A', 'Diagnosis B'].\n"
    "Now, from the text below, extract exactly two conditions as shown.\n"
    "If a diagnosis is not a disease (e.g., 'normal sinus rhythm'), still include it."
    "If none found, return: [\"None\"].\n\n"
    "Text: {medical_text}"
)

extract_chain = LLMChain(llm=llm, prompt=extract_prompt, output_key="diagnosis")

translate_prompt = ChatPromptTemplate.from_template(
    "You’re a friendly and knowledgeable health assistant. Explain the following diagnosis in clear, human terms (as if talking to a patient who's not a doctor).\n\n### Diagnosis:\n{diagnosis}"
)
translate_chain = LLMChain(llm=llm, prompt=translate_prompt, output_key="explanation")

recommend_prompt = ChatPromptTemplate.from_template(
    "Based on the diagnosis, recommend medically sound lifestyle changes.\n\n### Diagnosis:\n{diagnosis}\n\nProvide 5–7 bullet points."
)
recommend_chain = LLMChain(llm=llm, prompt=recommend_prompt, output_key="recommendations")

medication_prompt = ChatPromptTemplate.from_template(
    "You are a cardiology assistant. Given the diagnosis, list common medications (by class or name if appropriate) "
    "that are used to manage this condition. If the condition is 'normal sinus rhythm', 'sinus rhythm', or 'NSR' or a normal, say "
    "'No medication is needed for this condition.' Always remind the user to consult a healthcare provider.\n\n"
    "### Diagnosis:\n{diagnosis}\n\nList 3–5 medications based on the diagnosis"
)

medication_chain = LLMChain(llm=llm, prompt=medication_prompt, output_key="medications")

# ─── Regex Matchers ───────────────────────────────────────────────────────────
RESEARCH_RE = re.compile(
    r"\b(search|study|drug|medication|prescription|pubmed|arxiv|reference|wiki|more info|paper|link|citation)\b",
    re.IGNORECASE
)
TREATMENT_RE = re.compile(
    r"\b(treat|resolve|manage|fix|cure|medication|medicine|drug|how can.*be (treated|resolved|cured)|problem|okay)\b",
    re.IGNORECASE
)
GREETING_RE = re.compile(
    r"""
    \b(
        hi+|               # hi, hii, hiii
        hey+|              # hey, heyy
        hello+|            # hello
        yo+|               # yo, yoo
        howdy|             # howdy
        greetings|         # greetings
        what's\s*up|       # what's up, whats up
        wassup|            # wassup, sup
        sup\b|             # sup
        good\s*(morning|afternoon|evening|day)|  # good morning etc.
        gm\b|              # gm (short for good morning)
        gn\b|              # gn (good night)
        hope\s+(you('re)?|ur)\s+(okay|good|well|doing well)|  # hope you're well
        how\s+(are|r)\s+(you|u)|    # how are you / how r u
        I'm\s+(good|okay|alright|fine)|  # I'm good etc.
        thank(s| you)?|    # thanks or thank you
        yo\b|
        heya|hiya|ahoy
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
    )

BENIGN_DIAGNOSES = {
    "normal sinus rhythm",
    "sinus rhythm",
    "nsr",
}

def is_benign_diagnosis(diagnosis: str, threshold: int = 85) -> bool:
    diagnosis = diagnosis.lower().strip()
    # Pick the closest match from the benign list
    match, score, _ = process.extractOne(diagnosis, BENIGN_DIAGNOSES, scorer=fuzz.ratio)
    return score >= threshold
# ─── High‑Risk Diagnoses & Symptom Map ───────────────────────────────────────
HIGH_RISK_DIAGNOSES = [
    "wpw syndrome", "atrial fibrillation", "ventricular tachycardia",
    "extreme tachycardia", "anteroseptal infarction", "inferior infarction",
    "myocardial infarction", "myocardial ischemia", "premature ventricular contraction",
    "premature atrial contraction", "pac", "left ventricular hypertrophy", "rvh",
    "st-t abnormality", "left anterior hemiblock", "atrioventricular block",
    "av block1", "clbbb", "crbb", "irbbb", "overload of left atrium",
    "abnormal heartbeat", "bradycardia"
]

symptom_map = {
    "wpw syndrome": "palpitations, dizziness, or chest discomfort",
    "atrial fibrillation": "irregular heartbeat, fatigue, shortness of breath, or dizziness",
    "atrial fibrlliration": "irregular heartbeat, fatigue, shortness of breath, or dizziness",  # typo fallback
    "ventricular tachycardia": "lightheadedness, rapid heartbeat, chest pain, or fainting",
    "extreme tachycardia": "chest discomfort, breathlessness, or a racing pulse",
    "anteroseptal infarction": "chest pain, shortness of breath, or sweating",
    "inferior infarction": "chest pain, nausea, or fainting",
    "myocardial infarction": "chest pain, pressure, nausea, or breathlessness",
    "history of myocardial infarction": "recurring chest discomfort or breathlessness during exertion",
    "myocardial ischemia": "chest tightness, shortness of breath, or jaw/arm pain",
    "premature ventricular contraction": "irregular heartbeats, skipped beats, or fluttering in the chest",
    "premature atrial contraction": "fluttering sensation, skipped beats, or mild chest discomfort",
    "pac": "mild palpitations or irregular heartbeat",
    "left ventricular hypertrophy": "shortness of breath, chest pain, or fatigue during activity",
    "rvh": "chest pain, fatigue, or fainting (especially during exertion)",
    "st-t abnormality": "dizziness, chest tightness, or signs of underlying ischemia",
    "left anterior hemiblock": "dizziness or syncope (if progressing)",
    "atrioventricular block": "slow heartbeat, dizziness, or fainting",
    "av block1": "usually no symptoms, but may cause fatigue or dizziness in some cases",
    "clbbb": "fatigue, exertional breathlessness, or chest pain",
    "crbb": "often asymptomatic but may include palpitations or dizziness",
    "irbbb": "usually no symptoms, but may indicate other conduction issues if present",
    "overload of left atrium": "shortness of breath, fatigue, or palpitations",
    "abnormal heartbeat": "palpitations, irregular pulse, or chest discomfort",
    "bradycardia": "fatigue, dizziness, or fainting, especially with exertion",
    "default": "any unusual symptoms like chest pain, dizziness, or shortness of breath"
    }

def detect_intent(msg):
    intent_prompt = f"""
    You are a classification assistant.
    Classify the message into one of: greeting, diagnosis, general_medical_question,
    casual_chat, research_request, follow_up, comparison.

    Definitions and rules:

    - "diagnosis":
      The user explicitly states or strongly implies that they themselves have a confirmed diagnosis 
      of a specific medical condition (past or present). 
      This includes phrases like "I was diagnosed with...", "my diagnosis is...", "the diagnosis was...", 
      or "what does it mean if I have [condition]?".
      Always choose "diagnosis" if the message contains both the idea of a diagnosis and the name of a condition, 
      even if the condition is also a general topic (e.g., hypertension, left ventricular hypertrophy).

    - "general_medical_question":
      The user is asking about symptoms, conditions, or heart health concepts without saying they 
      have been officially diagnosed. 
      This can include hypothetical questions, educational inquiries, causes, prevention, or risk factors.
      If the condition is mentioned but there is no personal diagnosis statement, choose this.

    - "follow_up":
      The user is asking something related to a recent topic or diagnosis, without fully repeating it.
      Examples: "will it need surgery?", "can it be cured?", "what about medication?", "is it dangerous?"

    - "comparison":
      The user is asking to compare two diagnoses, conditions, or test results.
      Examples: "Compare atrial fibrillation and sinus tachycardia", "What’s the difference between WPW and AVNRT?"

    Respond with only the category name.

    Message: "{msg}"
    """

    small_llm = ChatOpenAI(
        api_key=api_key,
        temperature=0.0,
        model="gpt-4o-mini"
    )
    intent = small_llm([HumanMessage(content=intent_prompt)]).content.strip().lower()
    return intent

extract_diag = LLMChain(llm=llm, prompt=extract_diag_prompt, output_key="diagnoses")
import json

def get_diagnoses_from_chain(chain_out):
    diag_a = None
    diag_b = None
    
    if not isinstance(chain_out, dict):
        return diag_a, diag_b
    
    if "diagnoses" in chain_out:
        diagnoses_val = chain_out["diagnoses"]
        
        # If it's a JSON string, parse it
        if isinstance(diagnoses_val, str):
            try:
                diagnoses_val = json.loads(diagnoses_val)
            except json.JSONDecodeError:
                diagnoses_val = [diagnoses_val]
        
        if isinstance(diagnoses_val, (list, tuple)):
            if len(diagnoses_val) >= 2:
                diag_a, diag_b = diagnoses_val[0].strip(), diagnoses_val[1].strip()
            elif len(diagnoses_val) == 1:
                diag_a = diagnoses_val[0].strip()
        else:
            diag_a = str(diagnoses_val).strip()
    
    return diag_a, diag_b


def get_last_diagnosis_from_memory(mem_vars):
    for h in reversed(mem_vars.get("history", [])):
        if isinstance(h, dict) and "last_diagnosis" in h:
            return h["last_diagnosis"]
    return None

import re
import pandas as pd
from typing import Optional

def clean_markdown_table(md_table: Optional[str], drop_notes: bool = True) -> str:
    """
    Convert a Markdown-style table (or an HTML <table> string) into cleaned HTML
    ready for insertion into the chat UI. If input is malformed or empty, return "".

    Accepts:
      - md_table: a markdown table string (| a | b |) OR a preformatted HTML table string.
    Returns:
      - HTML string, or "" if nothing usable.
    """
    if not md_table or not isinstance(md_table, str):
        return ""

    s = md_table.strip()

    # If the input already looks like an HTML table, return it directly (trimmed)
    if "<table" in s and ("</table>" in s or s.endswith(">")):
        return s

    # Extract only markdown table lines (start with '|' and have at least one more '|')
    table_lines = [line.rstrip() for line in s.splitlines() if line.strip().startswith("|") and "|" in line]

    # If no markdown table lines found, return empty string (avoid raising).
    if len(table_lines) < 2:
        return ""

    # Remove header alignment rows like |---|:---:| etc.
    table_lines = [
        line for line in table_lines
        if not re.match(r'^\|\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$', line.strip())
    ]

    # After removing alignment, still need at least header + one row
    if len(table_lines) < 2:
        return ""

    # Parse rows into cells. Skip malformed rows that don't have same cell count as header.
    try:
        data = [[cell.strip() for cell in row.split("|")[1:-1]] for row in table_lines]
    except Exception:
        return ""

    # Ensure consistent column counts
    header = data[0]
    if not header or len(header) == 0:
        return ""

    rows = [r for r in data[1:] if len(r) == len(header)]
    if not rows:
        # no valid rows
        return ""

    # Build DataFrame
    df = pd.DataFrame(rows, columns=header)

    if drop_notes and "Notes" in df.columns:
        df = df.drop(columns=["Notes"])

    # Strip extra whitespace inside cells
    df = df.applymap(lambda x: re.sub(r'\s+', ' ', x.strip()) if isinstance(x, str) else x)

    # Return table HTML with your class for styling
    table_html = df.to_html(classes="styled-table", index=False, border=0, escape=False)
    return table_html


# ─── Unified Response Function ────────────────────────────────────────────────
def full_response(message: str) -> dict:
    msg = message.strip()
    intent = detect_intent(msg)
    print(intent)

    # 1) Greeting
    if intent == "greeting":
        greeting_prompt = (
            "You are a friendly assistant. If the user greets you (e.g., says hello, hi, hey, good morning, "
            "good afternoon, good evening), respond warmly and naturally. Most of the time, include an emoji "
            "or smiley face 🙂, but do not use an emoji if it would feel forced or unnatural in the context. "
            "Do not treat messages like 'thank you', 'thanks', or 'goodbye' as greetings — instead, respond politely "
            "but without calling it a greeting."
            "After greeting the user, follow up by asking something like: "
            "'How are you doing today? Do you have any medical questions about ECG or heart health?' "
            "This follow-up should feel casual and inviting."
        )

        mem_vars = chat_memory.load_memory_variables({})
        history = mem_vars.get("history",[])
        msgs = [SystemMessage(content=greeting_prompt)] + history + [HumanMessage(content=msg)]
        reply = llm(messages=msgs).content
        chat_memory.save_context({"input":msg},
                                 {"output":reply})
        return {
            "ai_reply": reply,
            "references": None
        }

    # 2) Casual chat
    if intent == "casual_chat":
        mem_vars = chat_memory.load_memory_variables({})
        history = mem_vars.get("history",[])
        msgs = [SystemMessage(content="You are a friendly cardiobt assistant. Reply casual and warmly. If it is not related to ECG or health questions kindly redirect and state that you respond to health related questions only.")] + history + [HumanMessage(content=msg)]
        reply = llm(messages=msgs).content
        chat_memory.save_context({"input":msg},
                                 {"output":reply})
        return {"ai_reply":reply,
                "references":None}

    # 3) General medical question
    if intent == "general_medical_question":
        mem_vars = chat_memory.load_memory_variables({})
        history = mem_vars.get("history", [])
        expert_general = (
            "You are a knowledgeable and compassionate medical expert. "
            "Provide clear, accurate, and empathetic answers to general heart-health questions. "
            "Base your responses on well-established evidence or medical guidelines. "
            "Use approachable language, organize your answer with a brief summary followed by details, "
            "and always encourage the user to consult a healthcare provider when appropriate."
            )
        msgs = [SystemMessage(content=expert_general)] + history + [HumanMessage(content=msg)]
        reply = llm(messages=msgs).content
        chat_memory.save_context({"input": msg}, {"output": reply})
        return {"ai_reply": reply, "references": None}

    # 4) Diagnosis path
    if intent == "diagnosis":
        diag_text = extract_chain({"medical_text": msg})["diagnosis"].strip()
        diag_clean = re.sub(r'[^a-z0-9\s]', '', diag_text.lower())

        # Benign short-circuit
        if is_benign_diagnosis(diag_clean):
            ai_reply = f"{diag_text} means your heart’s electrical activity is normal. No treatment needed—just maintain a healthy lifestyle! 😊"
            chat_memory.save_context(
                {"input": msg, "last_diagnosis": diag_text},
                {"output": ai_reply}
                )
            return {
                "diagnosis": diag_text,
                "ai_reply": ai_reply,
                "references": None
            }

        # 6) Full advice + meds branch
        chains = SequentialChain(
            chains=[translate_chain, recommend_chain, medication_chain],
            input_variables=["diagnosis"],
            output_variables=["explanation", "recommendations", "medications"]
        )
        results = chains({"diagnosis": diag_text})
        explanation = results["explanation"]
        recommendations = results["recommendations"]
        medications = results["medications"]

        # Build the reply
        ai_reply = (
            f"{explanation}\n\n"
            "Here are some medically sound lifestyle tips:\n"
            f"{recommendations}\n\n"
            "💊 Common medications or drug classes used to manage this condition:\n"
            f"{medications}\n\n"
            "Please consult a healthcare provider before starting any medication."
        )

        # Add high‑risk red-flag check
        if any(hr in diag_clean for hr in HIGH_RISK_DIAGNOSES):
            symptoms = symptom_map.get(diag_clean, symptom_map["default"])
            ai_reply += (
                f"\n\n🩺 Do you currently have symptoms such as {symptoms}?"
                "\n\n🚨 If you're experiencing severe chest pain, fainting, or difficulty breathing, "
                "call your local emergency number immediately."
            )
        chat_memory.save_context(
            {"input": msg, "last_diagnosis": diag_text},
            {"output": ai_reply}
            )

        return {"diagnosis": diag_text, "ai_reply": ai_reply, "references": None}
    # Follow-up intent
    if intent == "follow_up":
        mem_vars = chat_memory.load_memory_variables({})
        last_diag = None
        # Retrieve last diagnosis from memory history
        for h in mem_vars.get("history", []):
            if isinstance(h, dict) and "last_diagnosis" in h:
                last_diag = h["last_diagnosis"]
        # If not found in structured way, try to find from last bot message
        if not last_diag:
            last_diag = "the condition we discussed earlier"

        followup_msg = f"This question refers to your earlier diagnosis: {last_diag}. {msg}"
        history = mem_vars.get("history", [])
        expert_context = (
            "You are a knowledgeable medical expert. "
            "Respond to the user’s follow-up question clearly and accurately, "
            "referring back to the previous diagnosis as medical context. "
            "Explain in simple, empathetic language. "
            "If offering suggestions, base them on well-established medical guidelines or reputable peer-reviewed evidence. "
            "Do not guess or offer a diagnosis. "
            "Remind the user to consult a healthcare provider when appropriate."
            )
        msgs = [SystemMessage(content=expert_context)] + history + [HumanMessage(content=followup_msg)]
        reply = llm(messages=msgs).content
        chat_memory.save_context({"input": msg}, {"output": reply})
        return {"ai_reply": reply, "references": None}
    if intent == "comparison":
        mem_vars = chat_memory.load_memory_variables({})

        # Run extraction
        chain_out = extract_diag({"medical_text": msg})
    
       

        if isinstance(chain_out.get("diagnoses"), str):
            try:
                chain_out["diagnoses"] = json.loads(chain_out["diagnoses"])
            except json.JSONDecodeError:
                pass
        diag_a, diag_b = get_diagnoses_from_chain(chain_out)
        # Only proceed if both diagnoses are found
        if not (diag_a and diag_b):
            ai_reply = (
                "I couldn't identify two diagnoses to compare. "
                "Please say something like: 'Compare atrial fibrillation vs sinus tachycardia'."
            )
            chat_memory.save_context({"input": msg}, {"output": ai_reply})
            return {"ai_reply": ai_reply, "references": None}

        # Build system and user prompts
        comparison_system = (
            "You are an expert cardiology assistant. The user asked to compare TWO clinical diagnoses. "
            "Produce ONLY a well-formed Markdown table (no extra prose except a 1-2 sentence summary after the table). "
            "Table columns must be: Feature, {A}, {B}, Notes. Use concise cells (1-2 short sentences). "
            "Include features relevant to clinical comparison (ECG findings, common symptoms, urgency, "
            "possible triggers, risk level, first-line management, common medications, likely outcome). "
            "If any feature is not applicable, use 'None'. "
            "After the table add a one-line safety/disclaimer: "
            "\"This comparison is informational only — consult a healthcare professional for personalised advice.\""
        ).format(A=diag_a, B=diag_b)

        user_prompt = (
            f"Compare the following diagnoses for the user as requested:\n\nDiagnosis A: {diag_a}\nDiagnosis B: {diag_b}\n\n"
            "Return only the Markdown table and then a single-line safety/disclaimer as instructed."
        )

        history = mem_vars.get("history", [])
        msgs = [SystemMessage(content=comparison_system)] + history + [HumanMessage(content=user_prompt)]
        reply = llm(messages=msgs).content
        clean_table = clean_markdown_table(reply)

        if not clean_table:
            ai_reply = (
                "I couldn't produce a nicely formatted table. Here's the raw output below."
            )
            chat_memory.save_context(
                {"input": msg, "last_comparison": f"{diag_a} vs {diag_b}"},
                {"output": ai_reply}
            )
            return {
                "ai_reply": ai_reply + "\n\n" + reply,
                "table_html": "",
                "comparison_table_raw": reply
            }

        # Success path — short text reply and HTML table
        comparison_templates = [
            "Comparison: {a} vs {b}",
            "{a} vs {b} — Comparison Table",
            "How {a} and {b} Differ",
            "Side-by-Side Comparison: {a} & {b}",
            "Key Differences Between {a} and {b}",
            "{a} Compared to {b}"
            ]
        ai_reply = random.choice(comparison_templates).format(a=diag_a,b=diag_b)
        chat_memory.save_context(
            {"input": msg, "last_comparison": f"{diag_a} vs {diag_b}"},
            {"output": ai_reply}
        )
        table_with_caption = f"<table class='styled-table'><caption>{ai_reply}</caption>{clean_table}</table>"

        return {
            "ai_reply": "",
            "table_html": table_with_caption,
            "references": None
        }



    if RESEARCH_RE.search(msg):
        
        research = agent.run(msg)
        return {"ai_reply": research, "references": research}

    # 6) Default fallback
    mem_vars = chat_memory.load_memory_variables({})
    history = mem_vars.get("history", [])
    msgs = [SystemMessage(content=SYSTEM)] + history + [HumanMessage(content=msg)]
    reply = llm(messages=msgs).content
    chat_memory.save_context({"input": msg}, {"output": reply})
    return {"ai_reply": reply, "references": None}


# ─── Demo/Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== FULL RESPONSE EXAMPLE ===")
    user_msg = "what's the comparison between normal sinus rythm and left ventricular hypertrophy"
    result = full_response(user_msg)
    print("\nAI Reply:\n", result.get("ai_reply"))
