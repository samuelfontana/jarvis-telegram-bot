import json
import os
import logging
from typing import List, Dict
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from duckduckgo_search import DDGS
from flask import Flask
import threading

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configurações via variáveis de ambiente
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "")
PORT = int(os.environ.get("PORT", 10000))

MEMORY_FILE = "/tmp/jarvis_memory.json"
LEARNED_FILE = "/tmp/jarvis_learned.json"

# Inicializa o cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

# Modelo a usar
MODEL = "gpt-5-nano"

# Prompt de Sistema do Jarvis
SYSTEM_PROMPT = """Você é Jarvis, um assistente pessoal altamente inteligente, educado e eficiente, inspirado no assistente de Tony Stark.

Sua personalidade:
- Profissional, prestativo e levemente formal, mas sempre amigável
- Chama o usuário de "senhor" ocasionalmente, no estilo Jarvis
- Respostas diretas e objetivas, sem enrolação
- Quando não sabe algo, admite honestamente

Você é um generalista e está aqui para ajudar o usuário em qualquer tarefa. Você APRENDE com cada interação:
- Quando o usuário te ensina algo, memorize e aplique nas próximas conversas
- Lembre-se das preferências, gostos e informações pessoais do usuário
- Adapte-se ao estilo de comunicação do usuário com o tempo

Sempre responda em português brasileiro.

Se o usuário pedir informações atuais, notícias, preços ou qualquer coisa que exija dados em tempo real, use a função de pesquisa na web."""


# ===== Flask server para manter o Render ativo =====
app = Flask(__name__)

@app.route('/')
def home():
    return "Jarvis está online! 🤖"

@app.route('/health')
def health():
    return "OK", 200

def run_flask():
    app.run(host='0.0.0.0', port=PORT)


# ===== Funções de memória =====
def load_memory() -> Dict[str, List[Dict[str, str]]]:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar memória: {e}")
    return {}


def save_memory(memory: Dict[str, List[Dict[str, str]]]):
    try:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Erro ao salvar memória: {e}")


def load_learned() -> Dict[str, List[str]]:
    if os.path.exists(LEARNED_FILE):
        try:
            with open(LEARNED_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar aprendizados: {e}")
    return {}


def save_learned(learned: Dict[str, List[str]]):
    try:
        with open(LEARNED_FILE, 'w', encoding='utf-8') as f:
            json.dump(learned, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Erro ao salvar aprendizados: {e}")


# ===== Pesquisa web =====
def search_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "Nenhum resultado encontrado na pesquisa."
            summary = "Resultados da pesquisa na web:\n\n"
            for r in results:
                summary += f"- {r['title']}: {r['body']}\n  Fonte: {r['href']}\n\n"
            return summary
    except Exception as e:
        logger.error(f"Erro na pesquisa web: {e}")
        return "Não consegui realizar a pesquisa no momento."


# ===== Geração de resposta =====
async def get_jarvis_response(chat_id: str, user_text: str) -> str:
    memory = load_memory()
    learned = load_learned()

    if chat_id not in memory:
        memory[chat_id] = []

    # Monta o prompt com aprendizados
    system_content = SYSTEM_PROMPT
    if chat_id in learned and learned[chat_id]:
        system_content += "\n\nINFORMAÇÕES QUE VOCÊ JÁ APRENDEU SOBRE ESTE USUÁRIO:\n"
        for info in learned[chat_id]:
            system_content += f"- {info}\n"

    # Monta mensagens com contexto (últimas 30 mensagens)
    messages = [{"role": "system", "content": system_content}]
    history = memory[chat_id][-30:]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    # Ferramentas
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Pesquisa na web por informações em tempo real. Use quando o usuário perguntar sobre notícias, preços, eventos atuais, ou qualquer informação que precise ser atualizada.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "O termo de pesquisa em português ou inglês."}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "learn_info",
                "description": "Salva uma informação importante sobre o usuário para lembrar no futuro. Use quando o usuário ensinar algo, compartilhar uma preferência, ou dar uma informação pessoal relevante.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "info": {"type": "string", "description": "A informação a ser memorizada sobre o usuário."}
                    },
                    "required": ["info"]
                }
            }
        }
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        if response_message.tool_calls:
            tool_calls_data = []
            for tc in response_message.tool_calls:
                tool_calls_data.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })

            messages.append({
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": tool_calls_data
            })

            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if func_name == "search_web":
                    result = search_web(args.get("query", ""))
                elif func_name == "learn_info":
                    if chat_id not in learned:
                        learned[chat_id] = []
                    learned[chat_id].append(args.get("info", ""))
                    save_learned(learned)
                    result = "Informação memorizada com sucesso."
                else:
                    result = "Função não reconhecida."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            final_response = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )
            content = final_response.choices[0].message.content or "Desculpe, não consegui formular uma resposta."
        else:
            content = response_message.content or "Desculpe, não consegui formular uma resposta."

        # Salva na memória
        memory[chat_id].append({"role": "user", "content": user_text})
        memory[chat_id].append({"role": "assistant", "content": content})

        if len(memory[chat_id]) > 100:
            memory[chat_id] = memory[chat_id][-100:]

        save_memory(memory)
        return content

    except Exception as e:
        logger.error(f"Erro ao gerar resposta: {e}")
        return f"Senhor, tive um problema técnico ao processar sua mensagem. Erro: {str(e)}"


# ===== Handlers do Telegram =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    welcome_msg = (
        f"Olá, {user.first_name}. Eu sou o Jarvis, seu assistente pessoal.\n\n"
        f"Estou aqui para ajudá-lo no que precisar. Posso pesquisar informações na internet, "
        f"lembrar de coisas que você me ensinar, e aprender com nossas conversas.\n\n"
        f"Comandos disponíveis:\n"
        f"/start - Mensagem de boas-vindas\n"
        f"/memoria - Ver o que eu já aprendi sobre você\n"
        f"/reset - Limpar histórico de conversas\n\n"
        f"Em que posso ser útil hoje?"
    )
    await update.message.reply_text(welcome_msg)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    chat_id = str(update.effective_chat.id)
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    response = await get_jarvis_response(chat_id, user_text)

    if len(response) > 4000:
        parts = [response[i:i+4000] for i in range(0, len(response), 4000)]
        for part in parts:
            await update.message.reply_text(part)
    else:
        await update.message.reply_text(response)


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    memory = load_memory()
    if chat_id in memory:
        memory[chat_id] = []
        save_memory(memory)
    await update.message.reply_text("Memória de conversas limpa, senhor. Começamos do zero. (Os aprendizados permanecem.)")


async def memory_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    learned = load_learned()
    if chat_id in learned and learned[chat_id]:
        msg = "Aqui está o que aprendi sobre você, senhor:\n\n"
        for i, info in enumerate(learned[chat_id], 1):
            msg += f"{i}. {info}\n"
    else:
        msg = "Ainda não aprendi nada específico sobre você, senhor. Me ensine coisas e eu vou lembrar!"
    await update.message.reply_text(msg)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Erro: {context.error}")


# ===== Main =====
if __name__ == '__main__':
    # Inicia o Flask em uma thread separada (mantém o Render ativo)
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info(f"Flask server iniciado na porta {PORT}")

    # Inicia o bot do Telegram
    from telegram.request import HTTPXRequest
    request = HTTPXRequest(connect_timeout=30, read_timeout=30)
    application = (
        ApplicationBuilder()
        .token(TOKEN)
        .request(request)
        .build()
    )

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('reset', reset))
    application.add_handler(CommandHandler('memoria', memory_cmd))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    application.add_error_handler(error_handler)

    logger.info("Jarvis está online e aguardando comandos...")
    print("Jarvis está online e aguardando comandos...")

    application.run_polling(drop_pending_updates=True)
