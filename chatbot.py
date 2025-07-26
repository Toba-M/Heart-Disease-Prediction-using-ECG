import os
import openai
import warnings
import re
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
    input_key="input",
    output_key="output",
    return_messages=True
)

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM = """
You are CardioBot, a friendly and precise assistant focused on heart health and ECG-related concerns.

✅ Always warmly acknowledge greetings like "Hi" or "How are you?" — respond briefly and politely before inviting a health-related question.

🫁 You specialize in helping users understand symptoms (e.g., fatigue, chest pain, palpitations), ECG results, heart conditions, and treatment options.

💊 If the user asks for advice on medications or anything relating to medical things, supplements, or treatments, treat it as a medical question.  
— Ask for more context if needed (e.g., symptoms, diagnosis).  
— Never recommend prescription drugs without understanding the user’s condition.  
— Encourage follow-up with a healthcare provider for personal medication guidance.

❌ If the user asks about non-medical topics (e.g., sports, movies, politics), gently redirect:
"I'm here to assist with your health concerns — especially your heart. Feel free to ask me anything medical."

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
    "Extract the medical condition or diagnosis from the following patient text.\n"
    "If there is none, just return EXACTLY the word None (no quotes).\n\n{medical_text}"
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
    "that are used to manage this condition. Always remind the user to consult a healthcare provider before using any medication.\n\n"
    "### Diagnosis:\n{diagnosis}\n\nList 3–5 medications or drug classes."
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
# ─── Benign Diagnoses to Short‑Circuit ────────────────────────────────────────
BENIGN_DIAGNOSES = {
    "normal sinus rhythm",
    "sinus rhythm",
    "nsr",
}

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

# ─── Unified Response Function ────────────────────────────────────────────────
def full_response(message: str) -> dict:
    msg = message.strip()
    if GREETING_RE.search(msg):
        return {
        "ai_reply": "Hi there! 😊 I’m here to help with anything related to your heart health or ECG. How can I assist you today?",
        "references": None
        }
    # 2) Extract diagnosis
    diag_text = extract_chain({"medical_text": msg})["diagnosis"].strip()
    diag_lower = diag_text.lower()

    # 3) Short‑circuit benign rhythms
    if diag_lower in BENIGN_DIAGNOSES:
        return {
            "diagnosis": diag_text,
            "ai_reply": (
                f"{diag_text} means your heart’s electrical activity is normal. "
                "No treatment needed—just maintain a healthy lifestyle! 😊"
            ),
            "references": None
        }

    # 4) No diagnosable condition
    if diag_lower == "none":
        return {
            "ai_reply": (
                "I didn’t detect a specific condition. "
                "Could you share your symptoms or ECG findings in more detail?"
            ),
            "references": None
        }
    IDLE_CHAT_RE = re.compile(r"\b(who are you|your name|what can you do|thank you|i'm fine|i'm good|cool|awesome)\b", re.IGNORECASE)

    if IDLE_CHAT_RE.search(msg):
        return {
            "ai_reply": "I'm CardioBot — your friendly assistant for anything heart-related 💓. Let me know if you want to understand an ECG result or have a heart-health question!",
            "references": None
        }
    # 5) Treatment‑style questions
    if TREATMENT_RE.search(msg):
        explanation = translate_chain({"diagnosis": diag_text})["explanation"]
        return {
            "diagnosis": diag_text,
            "ai_reply": explanation,
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
        f"{diag_text} is a condition characterized as follows:\n{explanation}\n\n"
        "Here are some medically sound lifestyle tips:\n"
        f"{recommendations}\n\n"
        "💊 Common medications or drug classes used to manage this condition:\n"
        f"{medications}\n\n"
        "Please consult a healthcare provider before starting any medication."
    )

    # Add high‑risk red-flag check
    if any(hr in diag_lower for hr in HIGH_RISK_DIAGNOSES):
        symptoms = symptom_map.get(diag_lower, symptom_map["default"])
        ai_reply += (
            f"\n\n🩺 Do you currently have symptoms such as {symptoms}?"
            "\n\n🚨 If you're experiencing severe chest pain, fainting, or difficulty breathing, "
            "call your local emergency number immediately."
        )

    return {"diagnosis": diag_text, "ai_reply": ai_reply, "references": None}

    # 7) Research fallback
    research_requested = RESEARCH_RE.search(msg)
    if research_requested:
        research = agent.run(msg)
        return {"ai_reply": research, "references": research}

    # 8) Default chit‑chat/ memory
    mem_vars = chat_memory.load_memory_variables({})
    history = mem_vars.get("history", [])
    msgs = [SystemMessage(content=SYSTEM)] + history + [HumanMessage(content=msg)]
    reply = llm(messages=msgs).content
    chat_memory.save_context({"input": msg}, {"output": reply})
    return {"ai_reply": reply, "references": None}


# ─── Demo/Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== FULL RESPONSE EXAMPLE ===")
    user_msg = "I got a diagnosis of normal sinus rhythm. Is that okay?"
    result = full_response(user_msg)
    print("\nAI Reply:\n", result.get("ai_reply"))
