
import streamlit as st
import pymongo
from google import genai
from google.genai import types

# =======================
# CONFIGURACIÓN
# =======================

GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("❌ Faltan GOOGLE_API_KEY o MONGODB_URI")
    st.stop()

MODELO_EMBEDDING = "gemini-embedding-2"
DIMENSIONES = 3072
NOMBRE_INDICE = "ai_agents_vectors_index"

# =======================
# CLIENTES (cacheados)
# =======================

@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=GOOGLE_API_KEY)


@st.cache_resource
def get_mongo_collection():
    client = pymongo.MongoClient(MONGODB_URI)

    db = client["pdf_embeddings_db"]

    return db["ai_agents_vectors"]


client_genai = get_genai_client()
collection = get_mongo_collection()

# =======================
# FUNCIONES
# =======================

def crear_embedding(texto: str):
    """
    Genera el embedding de la consulta usando
    gemini-embedding-2 con 3072 dimensiones.
    """

    contenido = f"""
Representa la siguiente consulta para realizar una búsqueda
semántica dentro de un documento sobre agentes de inteligencia
artificial, adopción, autorización, confianza y escalamiento.

Consulta:

{texto}
"""

    response = client_genai.models.embed_content(
        model=MODELO_EMBEDDING,
        contents=contenido,
        config=types.EmbedContentConfig(
            output_dimensionality=DIMENSIONES
        ),
    )

    return response.embeddings[0].values


def buscar_similares(embedding, k=5):
    """
    Recupera los fragmentos más similares desde MongoDB Atlas.
    """

    pipeline = [
        {
            "$vectorSearch": {
                "index": NOMBRE_INDICE,
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "texto": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                },
            }
        },
    ]

    return list(collection.aggregate(pipeline))


def generar_respuesta(
    pregunta: str,
    contextos: list[dict]
) -> str:

    contexto = "\n\n".join(
        [c["texto"] for c in contextos]
    )

    prompt = f"""
Eres un asistente especializado en el informe
"AI Agents in Action: A Playbook for Trusted Adoption,
Authorization and Scaling" del World Economic Forum.

Responde EXCLUSIVAMENTE utilizando la información contenida
en el contexto recuperado.

El documento original está en inglés, pero debes responder
siempre de forma clara y concisa en español.

Si la respuesta no se encuentra en el contexto,
indícalo claramente y no inventes información.

Contexto:
{contexto}

Pregunta:
{pregunta}

Respuesta:
"""

    response = client_genai.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )

    return response.text


# =======================
# INTERFAZ STREAMLIT
# =======================

st.set_page_config(
    page_title="AI Agents RAG",
    page_icon="🤖"
)

st.title("🤖 Asistente RAG sobre Agentes de IA")

st.caption(
    "Consulta el informe AI Agents in Action del World Economic Forum"
)

if "historial" not in st.session_state:
    st.session_state.historial = []


# Mostrar historial
for msg in st.session_state.historial:

    if msg["rol"] == "usuario":

        st.chat_message("user").write(
            msg["texto"]
        )

    else:

        st.chat_message("assistant").write(
            msg["texto"]
        )


pregunta = st.chat_input(
    "Pregunta algo sobre agentes de IA..."
)


if pregunta:

    st.chat_message("user").write(pregunta)

    st.session_state.historial.append({
        "rol": "usuario",
        "texto": pregunta
    })

    similares = []

    with st.chat_message("assistant"):

        with st.spinner(
            "Buscando información en el documento..."
        ):

            try:

                emb = crear_embedding(pregunta)

                similares = buscar_similares(
                    emb,
                    k=5
                )

                if not similares:

                    respuesta = (
                        "No encontré información relevante "
                        "en el documento."
                    )

                else:

                    respuesta = generar_respuesta(
                        pregunta,
                        similares
                    )

            except Exception as e:

                respuesta = (
                    f"⚠️ Ocurrió un error: {e}"
                )

        st.write(respuesta)

        # Mostrar los fragmentos recuperados
        if similares:

            with st.expander(
                "🔍 Fragmentos recuperados"
            ):

                for i, c in enumerate(
                    similares,
                    1
                ):

                    st.markdown(
                        f"**Fragmento {i}** "
                        f"— score: `{c['score']:.4f}`"
                    )

                    st.write(
                        c["texto"][:500]
                        + (
                            "…"
                            if len(c["texto"]) > 500
                            else ""
                        )
                    )

                    st.divider()


    st.session_state.historial.append({
        "rol": "bot",
        "texto": respuesta
    })
