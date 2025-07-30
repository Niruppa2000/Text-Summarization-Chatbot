import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from dotenv import load_dotenv
from featherless import Featherless
import os

# Load environment variables (optional for local testing)
load_dotenv()

# 🌐 Streamlit App Setup
st.set_page_config(page_title="LangChain Summarizer", page_icon="🦜")
st.title("🦜 LangChain: Summarize YouTube or Website Content")
st.subheader("Summarize content from a URL (YouTube or article)")

# 🔑 Sidebar for API Key
with st.sidebar:
    st.markdown("## 🔐 Featherless API Key")
    featherless_api_key = st.text_input("Enter FEATHERLESS_KEY", type="password")

# ✅ Initialize Featherless LLM
llm = None
if featherless_api_key:
    try:
        client = Featherless(api_key=featherless_api_key)

        from langchain_core.language_models.chat_models import ChatGeneration
        from langchain_core.messages import HumanMessage, SystemMessage

        class FeatherlessChatLLM:
            def __init__(self, client, model="featherless-ai/summarizer"):
                self.client = client
                self.model = model

            def invoke(self, prompt):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content

        llm = FeatherlessChatLLM(client)

    except Exception as e:
        st.sidebar.error(f"⚠️ Failed to initialize Featherless: {e}")

# 🌐 User URL input
generic_url = st.text_input("Enter YouTube or Web Article URL")

# 🧾 Prompt Template
prompt_template = """
You are an expert summarizer. Provide a clear and concise 300-word summary of the following content:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# 🚀 Summarization Logic
if st.button("📝 Summarize"):
    if not featherless_api_key.strip() or not generic_url.strip():
        st.error("Please provide both the API key and a valid URL.")
    elif not validators.url(generic_url):
        st.error("Invalid URL. Please enter a proper YouTube or website link.")
    elif not llm:
        st.error("LLM not initialized. Check your API key.")
    else:
        try:
            with st.spinner("⏳ Loading content and summarizing..."):
                # Load documents
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=True,
                        headers={
                            "User-Agent": "Mozilla/5.0"
                        }
                    )
                docs = loader.load()

                # Create summarization chain
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                summary = chain.invoke(docs)

                # Display result
                st.success("✅ Summary:")
                st.write(summary)

        except Exception as e:
            st.error("❌ Error occurred:")
            st.exception(e)

