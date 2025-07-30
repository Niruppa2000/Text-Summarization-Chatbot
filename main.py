import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from dotenv import load_dotenv
import os

# Optional: Load API key from .env (for local dev)
load_dotenv()

# 🌐 Streamlit App
st.set_page_config(page_title="LangChain: Summarize Text From YouTube or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YouTube or Website")
st.subheader("Summarize URL")

# 🔐 Get Groq API Key and URL
with st.sidebar:
    groq_api_key = st.text_input("🔑 GROQ_API_KEY", type="password")

# 🧠 LLM Initialization (inside if to prevent errors)
llm = None
if groq_api_key:
    try:
        llm = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key)  # 🔁 Fixed: use `api_key`, not `groq_api_key`
    except Exception as e:
        st.sidebar.error(f"Error initializing Groq LLM: {e}")

# 🌍 Get the URL
generic_url = st.text_input("Enter YouTube or Website URL", label_visibility="visible")

# 🧾 Prompt Template
prompt_template = """
You are an expert summarizer. Provide a clear and concise 300-word summary of the following content:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# 🧠 Process Button
if st.button("📝 Summarize"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both the API key and a valid URL.")
    elif not validators.url(generic_url):
        st.error("Invalid URL. Please enter a proper YouTube or website link.")
    elif not llm:
        st.error("LLM not initialized. Check your API key.")
    else:
        try:
            with st.spinner("⏳ Summarizing content..."):
                # 🧲 Load documents
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=True,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        }
                    )
                docs = loader.load()

                # 🔗 Summarization Chain
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                summary = chain.invoke(docs)

                # ✅ Output
                st.success("✅ Summary Generated:")
                st.write(summary)
        except Exception as e:
            st.error("❌ An error occurred:")
            st.exception(e)
